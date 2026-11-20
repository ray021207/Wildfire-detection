"""Phase 7a: build an error gallery for the chosen model at its operating point.

False alarms: every false alert EPISODE in the test splits (first alerting frame
before t=0). If there are more than MAX_ITEMS, a random sample (fixed seed) is
drawn, so category counts are not biased toward the most confident mistakes.
Misses: every test sequence not detected within 10 minutes of first visible smoke.

Each item gets an id. The gallery images (figures/p7_*.png) are then reviewed
by eye and each id is assigned a category in results/p7_error_labels.csv;
scripts/p7_summary.py turns that file into counts.

Usage: python scripts/p7_gallery.py --model p5_B2_finetune_diff+persist2of3 --op fa_budget
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import DATA, FIGURES, SEED, read_result  # noqa: E402
from src.evaluate import SCORES, load_scores  # noqa: E402
from src.imaging import TILE, TILE_XY, crop_offset, load_frame  # noqa: E402
from src.metrics import persistence  # noqa: E402

MAX_ITEMS = 48
PER_FIG = 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--op", default="fa_budget")
    args = ap.parse_args()
    base = args.model.split("+")[0]
    frames = pd.read_csv(SCORES / "frame_scores.csv")
    tiles = load_scores(base)
    p6 = read_result("p6_operational")["operating_points"][args.model]
    op = args.op if args.op in p6 else "f1"
    thr = p6[op]["threshold"]
    # The "+persistK ofN" score column already contains the K-of-N rule (phase 5
    # turns it into a score), so a plain threshold reproduces phase 6 exactly.
    k, n = (1, 1)
    boxes = pd.read_csv(DATA / "boxes.csv")
    idx = pd.read_csv(DATA / "frame_index.csv")[["sequence", "file", "orig_w", "orig_h"]]
    fa, miss = [], []
    test = frames[frames.split.str.startswith("test")]
    for seq, d in test.sort_values("timestamp").groupby("sequence"):
        a = persistence(d[args.model].values >= thr, k, n)
        starts = a & ~np.r_[False, a[:-1]]
        for i in np.where(starts & (d.offset_s.values < 0))[0]:
            r = d.iloc[i]
            fa.append({"kind": "false_alarm", "sequence": seq, "file": r.file, "offset_min": r.offset_s / 60,
                       "split": r.split, "night": bool(r.night), "row": int(r.name)})
        hit = np.where(a & (d.offset_s.values >= 0))[0]
        ttd = d.offset_s.values[hit[0]] / 60 if len(hit) else np.nan
        if not (ttd <= 10):
            pos = d[d.offset_s >= 0]
            r = pos.iloc[(pos.offset_s - 600).abs().argmin()]
            miss.append({"kind": "miss", "sequence": seq, "file": r.file, "offset_min": r.offset_s / 60,
                         "split": r.split, "night": bool(r.night), "row": int(r.name),
                         "ttd_min": None if np.isnan(ttd) else float(ttd)})
    rng = np.random.default_rng(SEED)
    n_fa_total = len(fa)
    if len(fa) > MAX_ITEMS:
        fa = [fa[i] for i in sorted(rng.choice(len(fa), MAX_ITEMS, replace=False))]
    items = pd.DataFrame(fa + miss)
    items.insert(0, "id", [f"{'F' if kd == 'false_alarm' else 'M'}{i:02d}" for i, kd in
                           zip(items.groupby("kind").cumcount(), items.kind)])
    items["top_tile"] = [int(np.argmax(tiles[r])) for r in items.row]
    items["top_tile_score"] = [float(tiles[r].max()) for r in items.row]
    items.to_csv(DATA / "p7_items.csv", index=False)
    print(f"{n_fa_total} false-alarm episodes ({len(fa)} shown), {len(miss)} misses; model {args.model} op {op} thr {thr:.4f}")
    for kind, prefix in (("false_alarm", "p7_false_alarms"), ("miss", "p7_misses")):
        sub = items[items.kind == kind]
        for g in range(0, len(sub), PER_FIG):
            draw(sub.iloc[g:g + PER_FIG], boxes, idx, FIGURES / f"{prefix}_{g // PER_FIG + 1}.png")


def draw(sub, boxes, idx, path):
    fig, ax = plt.subplots(len(sub), 2, figsize=(13, 2.9 * len(sub)), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = np.atleast_2d(ax)
    for j, (_, it) in enumerate(sub.iterrows()):
        im = load_frame(it.sequence, it.file)
        x, y = TILE_XY[it.top_tile]
        ax[j, 0].imshow(im)
        ax[j, 0].add_patch(patches.Rectangle((x, y), TILE, TILE, fill=False, ec="red", lw=2))
        bb = boxes[(boxes.sequence == it.sequence) & (boxes.file == it.file)]
        if len(bb):
            o = idx[(idx.sequence == it.sequence) & (idx.file == it.file)].iloc[0]
            s, dy = 1536 / o.orig_w, crop_offset(o.orig_w, o.orig_h)
            for _, b in bb.iterrows():
                ax[j, 0].add_patch(patches.Rectangle((b.x0 * s, b.y0 * s - dy), (b.x1 - b.x0) * s, (b.y1 - b.y0) * s,
                                                     fill=False, ec="cyan", lw=1.5))
        extra = f" TTD={it.ttd_min:.0f}m" if it.kind == "miss" and pd.notna(it.get("ttd_min")) else (
            " never detected" if it.kind == "miss" else "")
        ax[j, 0].set_title(f"{it.id}  {it.sequence}  {it.offset_min:+.0f} min{extra}", fontsize=8, loc="left")
        ax[j, 1].imshow(im[y:y + TILE, x:x + TILE])
        ax[j, 1].set_title(f"highest tile score {it.top_tile_score:.2f}", fontsize=8)
        for a in ax[j]:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


if __name__ == "__main__":
    main()
