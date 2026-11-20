"""Shared evaluation for every model: given tile scores for every frame,
report frame- and tile-level metrics on the test splits.

Threshold: chosen on the VALIDATION split (max frame-level F1), then frozen
and applied unchanged to the test splits. Choosing it on test would be cheating.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import DATA, read_result
from .imaging import N_TILES, box_to_tile_labels
from .metrics import best_f1_threshold, metrics_with_ci

SCORES = DATA / "scores"
TEST_SPLITS = ("test_seen", "test_unseen")


def save_scores(name: str, tile_scores: np.ndarray):
    SCORES.mkdir(parents=True, exist_ok=True)
    np.save(SCORES / f"{name}.npy", tile_scores.astype(np.float32))


def load_scores(name: str) -> np.ndarray:
    return np.load(SCORES / f"{name}.npy")


def tile_label_table(df: pd.DataFrame) -> dict:
    """{frame row index: 40 tile labels} for frames we can label at tile level:
    every negative frame of a box sequence (all tiles 0) and every boxed frame."""
    boxes = pd.read_csv(DATA / "boxes.csv")
    # Boxes were carried back to t=0, so before the smoke is measurably visible
    # they mark where it WILL be. Only use boxed frames from the measured onset on.
    onset = {p["sequence"]: p.get("measurable_onset_min")
             for p in read_result("p2_splits")["label_noise"]["box_sequences"]}
    out = {}
    box_seqs = set(boxes.sequence)
    by_frame = {k: g for k, g in boxes.groupby(["sequence", "file"])}
    for i, r in df[df.sequence.isin(box_seqs)].iterrows():
        if r.offset_s < 0:
            out[i] = np.zeros(N_TILES, np.int8)
        elif (r.sequence, r.file) in by_frame and onset.get(r.sequence) is not None \
                and r.offset_s / 60 >= onset[r.sequence]:
            lab = np.zeros(N_TILES, np.int8)
            for _, b in by_frame[(r.sequence, r.file)].iterrows():
                lab |= box_to_tile_labels((b.x0, b.y0, b.x1, b.y1), r.orig_w, r.orig_h)
            out[i] = lab
    return out


def evaluate(df: pd.DataFrame, tile_scores: np.ndarray, frame_scores: np.ndarray | None = None) -> dict:
    """df rows must be aligned with tile_scores rows. Returns a dict for results/*.json."""
    df = df.reset_index(drop=True)
    s = tile_scores.max(1) if frame_scores is None else frame_scores
    y = df["label"].values
    val = (df.split == "val").values
    thr = best_f1_threshold(y[val], s[val])
    out = {"threshold_from_val": thr,
           "val_frame": metrics_with_ci(y[val], s[val], df.sequence.values[val], thr, n_boot=200)}
    for sp in TEST_SPLITS:
        m = (df.split == sp).values
        if m.any():
            out[f"{sp}_frame"] = metrics_with_ci(y[m], s[m], df.sequence.values[m], thr)
    m = df.split.str.startswith("test").values
    out["test_all_frame"] = metrics_with_ci(y[m], s[m], df.sequence.values[m], thr)

    labs = tile_label_table(df)
    idx = [i for i in labs if df.split[i].startswith("test")]
    if idx:
        yt = np.concatenate([labs[i] for i in idx])
        st = np.concatenate([tile_scores[i] for i in idx])
        gt = np.repeat(df.sequence.values[idx], N_TILES)
        out["test_tile_boxseqs"] = metrics_with_ci(yt, st, gt, thr)
    return out
