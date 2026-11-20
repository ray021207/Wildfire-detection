"""Phase 5: compare single-frame and temporal models.

Inputs: tile scores already written by
  p4_A1_frozen_lr, p4_A2_finetune               (single frame)
  p5_B1_frozen_lr_diff, p5_B1_frozen_lr_diff_noalign, p5_B2_finetune_diff,
  p5_B2w_finetune_diff_warm (B2 started from A2's weights)  (learned temporal)
Adds the persistence rule (K of last N frames) on top of every model; (K, N)
is chosen on VALIDATION frame PR-AUC.

Writes data/scores/frame_scores.csv (one column per model variant, used in
phase 6) and results/p5_temporal.json.
Usage: python scripts/p5_temporal.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import FEATS, read_result, write_result  # noqa: E402
from src.evaluate import SCORES, evaluate, load_scores  # noqa: E402
from src.metrics import persistence_score  # noqa: E402

BASE_MODELS = ["p4_A1_frozen_lr", "p4_A2_finetune", "p5_B1_frozen_lr_diff",
               "p5_B1_frozen_lr_diff_noalign", "p5_B2_finetune_diff",
               "p5_B2w_finetune_diff_warm"]
KN_OPTIONS = [(2, 2), (2, 3), (3, 3), (3, 5)]


def main():
    df = pd.read_csv(FEATS / "frames.csv")
    frames = df[["sequence", "file", "timestamp", "offset_s", "label", "split", "night"]].copy()
    table = {}
    for m in BASE_MODELS:
        if not (SCORES / f"{m}.npy").exists():
            print(f"skip {m}: no scores")
            continue
        tiles = load_scores(m)
        frames[m] = tiles.max(1)
        r = read_result(m)
        table[m] = {k: r[k] for k in ("test_seen_frame", "test_unseen_frame", "test_all_frame", "val_frame") if k in r}
        if "test_tile_boxseqs" in r:
            table[m]["test_tile_boxseqs"] = r["test_tile_boxseqs"]
        # persistence on top of this model
        val = frames.split == "val"
        best = None
        kn_search = {}
        for k, n in KN_OPTIONS:
            ps = persistence_score(frames, m, k, n)
            ap = float(average_precision_score(frames.label[val], ps[val.values]))
            kn_search[f"{k}of{n}"] = ap
            if best is None or ap > best[0]:
                best = (ap, k, n, ps)
        _, k, n, ps = best
        name = f"{m}+persist{k}of{n}"
        frames[name] = ps
        res = evaluate(df, tiles, frame_scores=ps)
        res.pop("test_tile_boxseqs", None)  # persistence acts on frames, tiles unchanged
        table[name] = {**{kk: res[kk] for kk in ("test_seen_frame", "test_unseen_frame", "test_all_frame", "val_frame")},
                       "kn_search_val_pr_auc": kn_search}
    frames.to_csv(SCORES / "frame_scores.csv", index=False)

    def ap(name, split):
        return table.get(name, {}).get(f"{split}_frame", {}).get("pr_auc")

    alignment = {sp: {"aligned": ap("p5_B1_frozen_lr_diff", sp), "not_aligned": ap("p5_B1_frozen_lr_diff_noalign", sp)}
                 for sp in ("val", "test_seen", "test_unseen")}
    write_result("p5_temporal", {"models": table, "alignment_effect_frame_pr_auc": alignment,
                                 "kn_options": KN_OPTIONS})


if __name__ == "__main__":
    main()
