"""Phase 4a: frozen MobileNetV3-Large embeddings for every tile of every frame.

Three inputs per tile:
  cur      - the tile itself (single-frame model A)
  diff_al  - difference image vs the ALIGNED previous frame (temporal model B)
  diff_raw - difference image vs the previous frame without alignment
             (only to measure whether alignment helps)

To fit the storage budget each 960-d embedding is reduced to 256 numbers with
PCA (fitted on training frames only) and stored as float16:
  data/features/<kind>.npy  shape (n_frames, 40, 256)
  data/features/frames.csv  row order of the arrays

Usage: python scripts/p4_features.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import FEATS, SEED, seed_everything, write_result  # noqa: E402
from src.data import frame_inputs, load_index  # noqa: E402
from src.imaging import N_TILES, difference_image, normalise  # noqa: E402
from src.models import Embedder  # noqa: E402

PCA_DIM = 256


class AllInputs(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        cur, prev_al = frame_inputs(row, temporal=True, align=True)
        _, prev_raw = frame_inputs(row, temporal=True, align=False)
        d_al = np.stack([difference_image(c, p) for c, p in zip(cur, prev_al)])
        d_raw = np.stack([difference_image(c, p) for c, p in zip(cur, prev_raw)])
        return torch.from_numpy(np.concatenate([normalise(cur), normalise(d_al), normalise(d_raw)]))


def main():
    seed_everything()
    FEATS.mkdir(parents=True, exist_ok=True)
    df = load_index()
    df.to_csv(FEATS / "frames.csv", index=False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Embedder().to(dev).eval()
    loader = DataLoader(AllInputs(df), batch_size=4, num_workers=8, persistent_workers=False)
    n = len(df)
    raw = {k: np.zeros((n, N_TILES, 960), np.float16) for k in ("cur", "diff_al", "diff_raw")}
    t0, i = time.time(), 0
    with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
        for x in loader:
            b = x.shape[0]
            e = model(x.view(-1, 3, 224, 224).to(dev, non_blocking=True)).float().cpu().numpy()
            e = e.reshape(b, 3, N_TILES, 960)
            for j, k in enumerate(("cur", "diff_al", "diff_raw")):
                raw[k][i:i + b] = e[:, j]
            i += b
            if i % 400 < b:
                print(f"{i}/{n} frames, {time.time() - t0:.0f}s", flush=True)
    secs = time.time() - t0
    train = (df["split"] == "train").values
    rng = np.random.default_rng(SEED)
    explained = {}
    for k in list(raw):
        arr = raw.pop(k)
        fit_rows = arr[train].reshape(-1, 960).astype(np.float32)
        fit_rows = fit_rows[rng.choice(len(fit_rows), min(60000, len(fit_rows)), replace=False)]
        pca = PCA(PCA_DIM, random_state=SEED).fit(fit_rows)
        explained[k] = float(pca.explained_variance_ratio_.sum())
        out = np.zeros((n, N_TILES, PCA_DIM), np.float16)
        for s in range(0, n, 500):
            out[s:s + 500] = pca.transform(arr[s:s + 500].reshape(-1, 960).astype(np.float32)).reshape(-1, N_TILES, PCA_DIM)
        np.save(FEATS / f"{k}.npy", out)
        del arr
    write_result("p4_features", {
        "backbone": "mobilenet_v3_large IMAGENET1K_V2 (torchvision)",
        "n_frames": n, "tiles_per_frame": N_TILES,
        "embedding_dim": 960, "pca_dim": PCA_DIM,
        "pca_explained_variance": explained,
        "extraction_seconds_gpu_fp16_3_inputs": secs,
        "device": dev,
    })


if __name__ == "__main__":
    main()
