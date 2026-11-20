"""Light fine-tune of MobileNetV3-Large on tiles with the multiple-instance loss.

  --mode single    : input = RGB tile (Baseline A2)
  --mode temporal  : input = RGB tile + difference-vs-aligned-previous tile,
                     stacked into 6 channels (Model B2, "early fusion")

The best epoch is picked on validation frame PR-AUC. Weights go to
data/models/<name>.pt, tile scores to data/scores/<name>.npy, metrics to
results/<name>.json.

Usage: python scripts/p4_finetune.py --name p4_A2_finetune --mode single
       python scripts/p4_finetune.py --name p5_B2_finetune_diff --mode temporal
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import MODELS, SEED, seed_everything, write_result  # noqa: E402
from src.data import FrameDataset, ThreadLoader, load_index, model_inputs  # noqa: E402
from src.evaluate import evaluate, load_scores, save_scores  # noqa: E402
from src.imaging import N_TILES  # noqa: E402
from src.models import TileClassifier, mil_loss  # noqa: E402


def score_frames(model, df, temporal, dev, bs=4):
    loader = ThreadLoader(FrameDataset(df, temporal, train=False), batch_size=bs)
    out = np.zeros((len(df), N_TILES), np.float32)
    model.eval()
    with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
        for x, _, idx in loader:
            p = torch.sigmoid(model(model_inputs(x, temporal, dev)).float()).view(-1, N_TILES)
            out[idx.numpy()] = p.cpu().numpy()
    return out


class Watchdog:
    """If no training step finishes for `limit` seconds, write every thread's
    stack to stderr (once per stall), so a hang shows where it is stuck."""

    def __init__(self, limit=120):
        import faulthandler
        import threading
        self.limit, self.last, self.fired = limit, time.time(), False
        self._fh = faulthandler
        threading.Thread(target=self._run, daemon=True).start()

    def tick(self):
        self.last, self.fired = time.time(), False

    def _run(self):
        while True:
            time.sleep(10)
            if not self.fired and time.time() - self.last > self.limit:
                print(f"WATCHDOG: no step for {time.time() - self.last:.0f}s, thread stacks follow",
                      file=sys.stderr, flush=True)
                self._fh.dump_traceback(file=sys.stderr, all_threads=True)
                self.fired = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", choices=["single", "temporal"], required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-frames", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--init-from", default=None,
                    help="start from another trained model's weights (e.g. p4_A2_finetune). For a "
                         "temporal model the difference-channel filters start at ZERO, so training "
                         "begins exactly at the single-frame model and only adds what time helps with")
    ap.add_argument("--pseudo", default=None,
                    help="name of a scored model (e.g. p4_A1_frozen_lr); its top-2 tiles in each smoke "
                         "frame become extra positive tile labels")
    args = ap.parse_args()
    seed_everything()
    torch.set_num_threads(4)  # leave CPU cores to the loader threads (see NOTES, failures log)
    temporal = args.mode == "temporal"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    df = load_index()
    df["row"] = np.arange(len(df))  # row in data/scores/*.npy
    tr, va = df[df.split == "train"], df[df.split == "val"]
    pseudo_pos = None
    if args.pseudo:
        ps = load_scores(args.pseudo)
        top = np.argsort(-ps, axis=1)[:, :2]
        pseudo_pos = np.zeros_like(ps, dtype=np.float32)
        np.put_along_axis(pseudo_pos, top, 1.0, axis=1)
        pseudo_pos[df.label.values == 0] = 0.0
        pseudo_pos = torch.from_numpy(pseudo_pos)
    model = TileClassifier(in_ch=6 if temporal else 3)
    if args.init_from:
        src = torch.load(MODELS / f"{args.init_from}.pt", map_location="cpu", weights_only=True)
        own = model.state_dict()
        k = "features.0.0.weight"
        if src[k].shape != own[k].shape:  # 3-channel source into 6-channel model
            w = torch.zeros_like(own[k])
            w[:, :3] = src[k]
            src[k] = w
        model.load_state_dict(src)
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * (len(tr) // args.batch_frames)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    train_ds = FrameDataset(tr, temporal, train=True, seed=SEED)
    loader = ThreadLoader(train_ds, batch_size=args.batch_frames, shuffle=True, seed=SEED, drop_last=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    best, history = (-1, None), []
    dog = Watchdog()
    t_start = time.time()
    for ep in range(args.epochs):
        model.train()
        t0, losses = time.time(), []
        train_ds.epoch = ep
        for x, y, idx in loader:
            with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                logits = model(model_inputs(x, temporal, dev))
            loss = mil_loss(logits.float(), y.to(dev), N_TILES)
            if pseudo_pos is not None:
                rows = torch.from_numpy(tr.row.values[idx.numpy()])
                tgt = pseudo_pos[rows].to(dev)            # (B, 40): 1 on pseudo-positive tiles
                lg = logits.float().view(-1, N_TILES)
                m = tgt > 0
                if m.any():
                    loss = loss + F.binary_cross_entropy_with_logits(lg[m], tgt[m])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
            dog.tick()
            if len(losses) % 50 == 0:
                print(f"  epoch {ep + 1} step {len(losses)} loss {np.mean(losses[-100:]):.3f} "
                      f"{time.time() - t0:.0f}s", flush=True)
        dog.limit = 3600  # validation scoring has no ticks
        sv = score_frames(model, va, temporal, dev).max(1)
        dog.limit = 120
        dog.tick()
        ap_val = float(average_precision_score(va.label, sv))
        history.append({"epoch": ep + 1, "train_loss": float(np.mean(losses)), "val_frame_pr_auc": ap_val,
                        "epoch_seconds": time.time() - t0})
        print(history[-1], flush=True)
        if ap_val > best[0]:
            best = (ap_val, ep + 1)
            torch.save(model.state_dict(), MODELS / f"{args.name}.pt")
    train_secs = time.time() - t_start
    model.load_state_dict(torch.load(MODELS / f"{args.name}.pt", map_location=dev, weights_only=True))
    scores = score_frames(model, df, temporal, dev)
    save_scores(args.name, scores)
    res = evaluate(df, scores)
    write_result(args.name, {"model": args.name, "mode": args.mode, "epochs": args.epochs,
                             "best_epoch": best[1], "history": history, "train_seconds": train_secs,
                             "lr": args.lr, "pseudo_labels_from": args.pseudo, "init_from": args.init_from, "batch_frames": args.batch_frames, "device": dev, **res})


if __name__ == "__main__":
    main()
