"""Phase 2: frame labels, label-noise analysis, leakage demo, and the splits.

Labels
  frame label = 1 if offset_s >= 0 (at/after FIgLib's "first visible smoke"), else 0.
  tile labels  = from HPWREN bounding boxes where they exist (9 sequences).

Splits (unit = sequence, grouping = fire DAY, so two cameras watching the same
fire, or two fires on the same day, never straddle train and test):
  1. Pick held-out camera SITES at random (fixed seed) until they hold ~18% of
     sequences. A site is a mountain top; holding out the whole site (all its
     directions and imagers) is stricter than holding out a single camera.
  2. Every fire day that touches a held-out site goes entirely to test.
  3. Fire days containing a bounding-box sequence go to test too, so tile-level
     metrics are always measured on data the model never trained on.
  4. Remaining fire days are split at random into train / val / test.
  A test sequence counts as "unseen camera" if its site never appears in
  train or val, otherwise "seen camera".

Usage: python scripts/p2_splits.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import DATA, FIGURES, FRAMES, SEED, write_result  # noqa: E402

HELDOUT_SITE_SHARE = 0.15
VAL_SHARE, TEST_SHARE = 0.15, 0.10  # of the remaining (non-forced) fire days


def thumbs(df, size=(32, 24)):
    out = np.zeros((len(df), size[0] * size[1]), dtype=np.float32)
    for i, (s, f) in enumerate(zip(df["sequence"], df["file"])):
        out[i] = np.asarray(Image.open(FRAMES / s / f).convert("L").resize(size), dtype=np.float32).ravel()
    return out


def leakage_demo(df, rng):
    """Why a random frame split leaks: neighbours in time are near-duplicates."""
    X = thumbs(df)
    seq = df["sequence"].values
    y = (df["offset_s"].values >= 0).astype(int)
    order = np.lexsort((df["timestamp"].values, seq))
    X, seq, y = X[order], seq[order], y[order]
    same_next = seq[1:] == seq[:-1]
    adj_mad = np.abs(X[1:] - X[:-1]).mean(1)[same_next]
    i = rng.integers(0, len(X), 4000)
    j = rng.integers(0, len(X), 4000)
    keep = seq[i] != seq[j]
    cross_mad = np.abs(X[i[keep]] - X[j[keep]]).mean(1)

    def one_nn(train_idx, test_idx):
        A, B = X[train_idx], X[test_idx]
        a2 = (A ** 2).sum(1)
        preds, same = [], []
        for k in range(0, len(B), 512):
            b = B[k:k + 512]
            d = a2[None, :] - 2 * b @ A.T
            nn = train_idx[d.argmin(1)]
            preds.append(y[nn])
            same.append(seq[nn] == seq[test_idx[k:k + 512]])
        return float((np.concatenate(preds) == y[test_idx]).mean()), float(np.concatenate(same).mean())

    n = len(X)
    perm = rng.permutation(n)
    tr, te = perm[: int(0.8 * n)], perm[int(0.8 * n):]
    acc_random, same_random = one_nn(np.sort(tr), np.sort(te))
    in_train = np.zeros(n, bool)
    in_train[tr] = True
    has_adj = np.zeros(n, bool)
    has_adj[1:] |= in_train[:-1] & (seq[1:] == seq[:-1])
    has_adj[:-1] |= in_train[1:] & (seq[1:] == seq[:-1])
    adj_share = float(has_adj[te].mean())

    useqs = np.unique(seq)
    test_seqs = set(rng.choice(useqs, int(0.2 * len(useqs)), replace=False))
    is_te = np.array([s in test_seqs for s in seq])
    acc_seq, same_seq = one_nn(np.where(~is_te)[0], np.where(is_te)[0])

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, max(np.percentile(cross_mad, 99), 1), 80)
    ax.hist(adj_mad, bins=bins, alpha=0.7, density=True, label="consecutive frames, same sequence")
    ax.hist(cross_mad, bins=bins, alpha=0.7, density=True, label="frames from different sequences")
    ax.set(xlabel="mean absolute grey-level difference (32x24 thumbnails)", ylabel="density",
           title="Adjacent frames are near-duplicates")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "p2_leakage.png", dpi=110)
    plt.close(fig)
    return {
        "adjacent_frame_mad_median": float(np.median(adj_mad)),
        "cross_sequence_mad_median": float(np.median(cross_mad)),
        "random_split_test_frames_with_adjacent_train_frame": adj_share,
        "one_nn_accuracy_random_frame_split": acc_random,
        "one_nn_neighbour_from_same_sequence_random_split": same_random,
        "one_nn_accuracy_sequence_split": acc_seq,
        "one_nn_neighbour_from_same_sequence_sequence_split": same_seq,
        "majority_class_accuracy": float(max(y.mean(), 1 - y.mean())),
    }


def label_noise(df, boxes):
    """How well does 'offset >= 0' match what a human actually boxed?"""
    out = {}
    if not len(boxes):
        return out
    meta = df.set_index(["sequence", "file"])
    b = boxes.merge(df[["sequence", "file", "offset_s", "orig_w", "orig_h"]], on=["sequence", "file"])
    b["area_frac"] = (b.x1 - b.x0) * (b.y1 - b.y0) / (b.orig_w * b.orig_h)
    b["minute"] = (b.offset_s // 60).astype(int)
    per_seq = []
    for s, d in df[df["sequence"].isin(boxes["sequence"])].groupby("sequence"):
        pos = d[d.offset_s >= 0]
        boxed = set(boxes[boxes.sequence == s].file)
        first_box = b[b.sequence == s].offset_s.min()
        per_seq.append({
            "sequence": s,
            "positive_frames": int(len(pos)),
            "positive_frames_without_box": int((~pos.file.isin(boxed)).sum()),
            "negative_frames_with_box": int(((d.offset_s < 0) & d.file.isin(boxed)).sum()),
            "first_boxed_offset_min": float(first_box / 60) if pd.notna(first_box) else None,
        })
    onset = measurable_onset(df, b)
    for p in per_seq:
        p.update(onset.get(p["sequence"], {}))
        p["n_distinct_boxes"] = int(b[b.sequence == p["sequence"]][["x0", "y0", "x1", "y1"]].drop_duplicates().shape[0])
    on = [p["measurable_onset_min"] for p in per_seq if p.get("measurable_onset_min") is not None]
    out["measurable_onset_min_median"] = float(np.median(on)) if on else None
    out["measurable_onset_min_values"] = on
    out["n_box_sequences_never_measurable"] = int(sum(p.get("measurable_onset_min") is None for p in per_seq))
    by_min = b[b.minute.between(0, 39)].groupby("minute")["area_frac"].median()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(by_min.index, by_min.values * 100, "o-")
    ax.set(xlabel="minutes after first visible smoke", ylabel="median box area (% of frame, log)",
           title="Smoke starts tiny: box size vs time (HPWREN boxes)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "p2_box_area_vs_time.png", dpi=110)
    plt.close(fig)
    early = b[b.offset_s.between(0, 5 * 60)]
    out.update({
        "box_sequences": per_seq,
        "box_area_pct_median_first_5min": float(early.area_frac.median() * 100) if len(early) else None,
        "box_area_pct_median_after_30min": float(b[b.offset_s >= 1800].area_frac.median() * 100),
        "box_area_pct_by_minute": {int(k): float(v * 100) for k, v in by_min.items()},
        "share_positive_frames_without_box": float(
            sum(p["positive_frames_without_box"] for p in per_seq) / max(1, sum(p["positive_frames"] for p in per_seq))),
    })
    return out


def measurable_onset(df, b):
    """When does the boxed region actually start to change?

    The HPWREN boxes were drawn on the big late plume and carried backwards to
    t=0, so they say WHERE smoke will be, not WHEN it becomes visible. For each
    box sequence we take the union of its boxes, build a smoke-free reference
    (median of the pre-t0 frames inside that region), and measure how far each
    frame departs from it. The noise level is the mean + 3 std of that distance
    over pre-t0 frames. "Measurable onset" = first offset >= 0 where the
    distance exceeds the noise level in two consecutive frames.
    """
    res = {}
    for seq, bb in b.groupby("sequence"):
        d = df[df.sequence == seq].sort_values("offset_s")
        s = 1536 / d.orig_w.iloc[0]
        x0, y0 = int(bb.x0.min() * s), int(bb.y0.min() * s)
        x1, y1 = int(bb.x1.max() * s), int(bb.y1.max() * s)
        crops = np.stack([np.asarray(Image.open(FRAMES / seq / f).convert("L"), dtype=np.float32)[y0:y1, x0:x1]
                          for f in d.file])
        pre = d.offset_s.values < 0
        ref = np.median(crops[pre], axis=0)
        dist = np.abs(crops - ref).mean(axis=(1, 2))
        noise = dist[pre].mean() + 3 * dist[pre].std()
        above = (dist > noise) & ~pre
        hits = np.where(above[:-1] & above[1:])[0]
        res[seq] = {"measurable_onset_min": float(d.offset_s.values[hits[0]] / 60) if len(hits) else None,
                    "noise_level": float(noise)}
    return res


def make_splits(seq_idx, box_seqs, rng):
    seq_idx = seq_idx.copy()
    seq_idx["day"] = seq_idx.index.str[:8]
    n = len(seq_idx)
    sites = sorted(seq_idx["site"].unique())
    rng.shuffle(sites)
    held = []
    for s in sites:
        if (seq_idx["site"].isin(held)).sum() >= HELDOUT_SITE_SHARE * n:
            break
        if seq_idx.index[seq_idx.site == s].isin(box_seqs).any():
            continue  # keep box sequences on seen cameras so tile metrics test "seen" conditions
        held.append(s)
    forced_days = set(seq_idx.loc[seq_idx.site.isin(held), "day"]) | set(seq_idx.loc[seq_idx.index.isin(box_seqs), "day"])
    free_days = sorted(set(seq_idx.day) - forced_days)
    rng.shuffle(free_days)
    n_val = int(round(VAL_SHARE * len(free_days)))
    n_test = int(round(TEST_SHARE * len(free_days)))
    val_days = set(free_days[:n_val])
    test_days = set(free_days[n_val:n_val + n_test]) | forced_days
    seq_idx["split"] = np.where(seq_idx.day.isin(val_days), "val",
                                np.where(seq_idx.day.isin(test_days), "test", "train"))
    seen_sites = set(seq_idx.loc[seq_idx.split.isin(["train", "val"]), "site"])
    is_test = seq_idx.split == "test"
    seq_idx.loc[is_test & seq_idx.site.isin(seen_sites), "split"] = "test_seen"
    seq_idx.loc[is_test & ~seq_idx.site.isin(seen_sites), "split"] = "test_unseen"
    return seq_idx, held


def main():
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(DATA / "frame_index.csv")
    seq_idx = pd.read_csv(DATA / "sequence_index.csv", index_col=0)
    boxes = pd.read_csv(DATA / "boxes.csv")
    df["label"] = (df["offset_s"] >= 0).astype(int)

    noise = label_noise(df, boxes)
    leak = leakage_demo(df, rng)
    splits, held = make_splits(seq_idx, set(boxes["sequence"]), rng)
    splits[["split", "site", "camera", "event", "day"]].to_csv(DATA / "splits.csv")
    df = df.merge(splits[["split"]], left_on="sequence", right_index=True)
    df.to_csv(DATA / "frame_index.csv", index=False)

    summary = {}
    for sp, d in df.groupby("split"):
        s = splits[splits.split == sp]
        summary[sp] = {"sequences": int(len(s)), "fire_days": int(s.day.nunique()),
                       "sites": sorted(s.site.unique().tolist()), "frames": int(len(d)),
                       "positive_frames": int(d.label.sum()), "negative_frames": int((1 - d.label).sum()),
                       "night_frames": int(d.night.sum())}
    # sanity checks: no fire day in two splits; unseen sites absent from train/val
    day_splits = splits.groupby("day")["split"].apply(lambda s: set(x.replace("_seen", "").replace("_unseen", "") for x in s))
    assert all(len(v) == 1 for v in day_splits), "fire day straddles splits"
    trval_sites = set(splits.loc[splits.split.isin(["train", "val"]), "site"])
    assert not (set(splits.loc[splits.split == "test_unseen", "site"]) & trval_sites)

    write_result("p2_splits", {
        "label_rule": "frame label = 1 if offset_s >= 0 (FIgLib first-visible-smoke time), else 0",
        "heldout_sites_chosen": held,
        "splits": summary,
        "sequences_by_split": {sp: sorted(splits.index[splits.split == sp].tolist()) for sp in summary},
        "label_noise": noise,
        "leakage": leak,
    })


if __name__ == "__main__":
    main()
