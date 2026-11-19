"""Phase 3: preprocessing, explained with figures and measured.

  1. Tiling vs whole-image downscaling   -> figures/p3_tiling.png
  2. Resize + ImageNet normalisation     -> figures/p3_normalisation.png
  3. Frame differencing + alignment      -> figures/p3_differencing.png, p3_diff_failures.png
     Alignment is measured on every consecutive pair of pre-smoke frames:
     does shifting the previous frame onto the current one reduce the
     "spurious change" that has nothing to do with smoke?
     Shifts are saved to data/shifts.csv and reused in phase 5.
  4. Augmentation examples               -> figures/p3_augmentation.png

Writes results/p3_preprocess.json.
Usage: python scripts/p3_preprocess.py
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import DATA, FIGURES, write_result  # noqa: E402
from src.imaging import (ALIGN_MIN_RESPONSE, ALIGN_MIN_SHIFT_PX, IMAGENET_MEAN, IMAGENET_STD, TILE, TILE_XY, apply_shift,  # noqa: E402
                         box_to_tile_labels, crop_offset, difference_image, estimate_shift, load_frame)
from src.augment import augment_tile  # noqa: E402

CHANGE_THRESH = 25  # grey levels; a pixel "changed" if |cur - prev| exceeds this


def pair_stats(args):
    seq, prev_f, cur_f = args
    prev, cur = load_frame(seq, prev_f), load_frame(seq, cur_f)
    dx, dy, resp = estimate_shift(prev, cur)
    g = lambda im: cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.int16)  # noqa: E731
    raw = np.abs(g(cur) - g(prev))
    aligned_prev = apply_shift(prev, dx, dy)
    ali = np.abs(g(cur) - g(aligned_prev))
    m = 16  # ignore a border where the shift replicated edge pixels
    raw, ali = raw[m:-m, m:-m], ali[m:-m, m:-m]
    gp, gc = g(prev).mean(), g(cur).mean()
    return {"sequence": seq, "file": cur_f, "prev_file": prev_f, "dx": dx, "dy": dy, "phase_resp": resp,
            "mad_raw": float(raw.mean()), "mad_aligned": float(ali.mean()),
            "changed_raw": float((raw > CHANGE_THRESH).mean()),
            "changed_aligned": float((ali > CHANGE_THRESH).mean()),
            "brightness_jump": float(abs(gc - gp))}


def compute_shifts(df):
    df = df.sort_values(["sequence", "timestamp"])
    jobs = []
    for seq, d in df.groupby("sequence"):
        files = d["file"].tolist()
        jobs += [(seq, a, b) for a, b in zip(files[:-1], files[1:])]
    with ThreadPoolExecutor(12) as ex:
        rows = list(ex.map(pair_stats, jobs, chunksize=16))
    out = pd.DataFrame(rows).merge(df[["sequence", "file", "offset_s", "night"]], on=["sequence", "file"])
    out.to_csv(DATA / "shifts.csv", index=False)
    return out


def fig_tiling(df, boxes):
    """Show the tile grid and why we do not just shrink the whole image."""
    b = boxes.merge(df[["sequence", "file", "offset_s", "orig_w", "orig_h"]], on=["sequence", "file"])
    b = b[b.offset_s.between(120, 360)].sort_values("offset_s").iloc[0]
    im = load_frame(b.sequence, b.file)
    lab = box_to_tile_labels((b.x0, b.y0, b.x1, b.y1), b.orig_w, b.orig_h)
    fig = plt.figure(figsize=(15, 7.5))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(im)
    for i, (x, y) in enumerate(TILE_XY):
        ax.add_patch(patches.Rectangle((x, y), TILE, TILE, fill=False, lw=1.6 if lab[i] else 0.6,
                                       ec="red" if lab[i] else "yellow"))
    s = 1536 / b.orig_w
    dy = crop_offset(b.orig_w, b.orig_h)
    ax.add_patch(patches.Rectangle((b.x0 * s, b.y0 * s - dy), (b.x1 - b.x0) * s, (b.y1 - b.y0) * s,
                                   fill=False, ec="cyan", lw=1.5))
    ax.set_title(f"{b.sequence} at +{b.offset_s / 60:.0f} min: 8x5 grid of 224 px tiles\n"
                 "red = tiles labelled smoke from the HPWREN box (cyan)", fontsize=9)
    ax.axis("off")
    i = int(np.argmax(lab))
    x, y = TILE_XY[i]
    ax2 = fig.add_subplot(2, 4, 3)
    ax2.imshow(im[y:y + TILE, x:x + TILE])
    ax2.set_title("one 224x224 tile (what the model sees)", fontsize=9)
    ax2.axis("off")
    small = cv2.resize(im, (224, 149), interpolation=cv2.INTER_AREA)
    ax3 = fig.add_subplot(2, 4, 4)
    ax3.imshow(small)
    ax3.set_title("whole frame shrunk to 224 px wide", fontsize=9)
    ax3.axis("off")
    sx = 224 / 1536
    ax4 = fig.add_subplot(2, 4, 8)
    ax4.imshow(cv2.resize(small[int(y * sx):int((y + TILE) * sx), int(x * sx):int((x + TILE) * sx)], (224, 224),
                          interpolation=cv2.INTER_NEAREST))
    ax4.set_title(f"same region after shrinking: {int(TILE * sx)}x{int(TILE * sx)} px", fontsize=9)
    ax4.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES / "p3_tiling.png", dpi=110)
    plt.close(fig)
    return {"tiling_example": f"{b.sequence}/{b.file}", "pixels_per_tile_after_whole_image_shrink": int(TILE * sx)}


def fig_normalisation(df):
    r = df.iloc[len(df) // 3]
    im = load_frame(r.sequence, r.file)
    t = im[TILE_XY[30][1]:TILE_XY[30][1] + TILE, TILE_XY[30][0]:TILE_XY[30][0] + TILE].astype(np.float32) / 255
    n = (t - IMAGENET_MEAN) / IMAGENET_STD
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].imshow(t)
    ax[0].set_title("tile, pixel values 0..1")
    ax[0].axis("off")
    for c, col in enumerate("rgb"):
        ax[1].hist(t[..., c].ravel(), bins=50, alpha=0.5, color=col)
        ax[2].hist(n[..., c].ravel(), bins=50, alpha=0.5, color=col)
    ax[1].set_title("before: value / 255")
    ax[2].set_title("after: (value - ImageNet mean) / ImageNet std")
    fig.tight_layout()
    fig.savefig(FIGURES / "p3_normalisation.png", dpi=110)
    plt.close(fig)


def fig_differencing(df, shifts, boxes):
    b = boxes.merge(df[["sequence", "file", "offset_s"]], on=["sequence", "file"])
    b = b[b.offset_s.between(120, 400)].sort_values("offset_s").iloc[0]
    sh = shifts[(shifts.sequence == b.sequence) & (shifts.file == b.file)].iloc[0]
    cur, prev = load_frame(b.sequence, b.file), load_frame(b.sequence, sh.prev_file)
    ali = apply_shift(prev, sh.dx, sh.dy)
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    for a, im, title in [(ax[0, 0], prev, "previous frame (t - 1 step)"), (ax[0, 1], cur, f"current frame (+{b.offset_s / 60:.0f} min)"),
                         (ax[1, 0], difference_image(cur, prev), "difference, no alignment"),
                         (ax[1, 1], difference_image(cur, ali), f"difference after shifting previous by ({sh.dx:.1f}, {sh.dy:.1f}) px")]:
        a.imshow(im)
        a.set_title(title, fontsize=10)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES / "p3_differencing.png", dpi=100)
    plt.close(fig)

    neg = shifts[shifts.offset_s < 0].assign(m=lambda d: np.hypot(d.dx, d.dy))
    conf = neg[neg.phase_resp >= ALIGN_MIN_RESPONSE]
    picks = {
        "camera shake (largest confident shift)": conf.sort_values("m").iloc[-1],
        "fooled estimator (large low-confidence shift)": neg[neg.phase_resp < ALIGN_MIN_RESPONSE].sort_values("m").iloc[-1],
        "lighting change (largest brightness jump)": neg.sort_values("brightness_jump").iloc[-1],
        "most change left after alignment (clouds, noise, ...)": neg[~neg.night].sort_values("changed_aligned").iloc[-1],
    }
    fig, ax = plt.subplots(len(picks), 3, figsize=(15, 3.6 * len(picks)))
    for row, (name, r) in enumerate(picks.items()):
        c, p = load_frame(r.sequence, r.file), load_frame(r.sequence, r.prev_file)
        ax[row, 0].imshow(c)
        ax[row, 0].set_title(f"{name}\n{r.sequence} {r.offset_s / 60:+.0f} min (no smoke)", fontsize=9)
        ax[row, 1].imshow(difference_image(c, p))
        ax[row, 1].set_title(f"raw difference: {r.changed_raw * 100:.1f}% pixels changed", fontsize=9)
        ax[row, 2].imshow(difference_image(c, apply_shift(p, r.dx, r.dy)))
        ax[row, 2].set_title(f"aligned: {r.changed_aligned * 100:.1f}% changed (shift {r.dx:.1f},{r.dy:.1f})", fontsize=9)
        for a in ax[row]:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES / "p3_diff_failures.png", dpi=90)
    plt.close(fig)
    return {k: f"{r.sequence}/{r.file}" for k, r in picks.items()}


def fig_augmentation(df, boxes):
    rng = np.random.default_rng(0)
    b = boxes.merge(df[["sequence", "file", "offset_s", "orig_w", "orig_h"]], on=["sequence", "file"])
    b = b[b.offset_s.between(300, 900)].iloc[0]
    im = load_frame(b.sequence, b.file)
    i = int(np.argmax(box_to_tile_labels((b.x0, b.y0, b.x1, b.y1), b.orig_w, b.orig_h)))
    x, y = TILE_XY[i]
    t = im[y:y + TILE, x:x + TILE]
    fig, ax = plt.subplots(1, 6, figsize=(16, 3))
    ax[0].imshow(t)
    ax[0].set_title("original tile")
    for k in range(1, 6):
        ax[k].imshow(augment_tile(t, rng))
        ax[k].set_title(f"augmented #{k}")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES / "p3_augmentation.png", dpi=110)
    plt.close(fig)


def gated_stats(neg):
    """Effect of alignment on (a) pairs that pass the confidence gate and
    (b) pairs with a big but low-confidence 'shift' (mostly clouds/lighting)."""
    m = np.hypot(neg.dx, neg.dy)
    passed = (neg.phase_resp >= ALIGN_MIN_RESPONSE) & (m >= ALIGN_MIN_SHIFT_PX)
    big_low = (neg.phase_resp < ALIGN_MIN_RESPONSE) & (m >= ALIGN_MIN_SHIFT_PX)
    out = {}
    for name, sel in (("pass_gate", passed), ("big_shift_low_confidence", big_low)):
        s = neg[sel]
        out[name] = {"n": int(sel.sum()),
                     "changed_raw_pct": float(s.changed_raw.mean() * 100) if len(s) else None,
                     "changed_aligned_pct": float(s.changed_aligned.mean() * 100) if len(s) else None,
                     "share_improved": float((s.changed_aligned < s.changed_raw).mean()) if len(s) else None}
    gated = np.where(passed, neg.changed_aligned, neg.changed_raw)
    out["all_neg_changed_pct_with_gated_policy"] = float(gated.mean() * 100)
    return out


def main():
    df = pd.read_csv(DATA / "frame_index.csv")
    boxes = pd.read_csv(DATA / "boxes.csv")
    shifts = compute_shifts(df)
    neg = shifts[shifts.offset_s < 0]
    mag = np.hypot(neg.dx, neg.dy)
    out = {
        "n_consecutive_pairs": int(len(shifts)),
        "n_negative_pairs": int(len(neg)),
        "shift_px_median": float(mag.median()),
        "shift_px_p95": float(mag.quantile(0.95)),
        "share_pairs_shift_over_1px": float((mag > 1).mean()),
        "share_pairs_shift_over_3px": float((mag > 3).mean()),
        "change_threshold_grey_levels": CHANGE_THRESH,
        "neg_changed_pixels_raw_mean_pct": float(neg.changed_raw.mean() * 100),
        "neg_changed_pixels_aligned_mean_pct": float(neg.changed_aligned.mean() * 100),
        "neg_mad_raw_mean": float(neg.mad_raw.mean()),
        "neg_mad_aligned_mean": float(neg.mad_aligned.mean()),
        "neg_share_pairs_alignment_reduced_change": float((neg.changed_aligned < neg.changed_raw).mean()),
        "low_confidence_pairs_response_below_gate": int((neg.phase_resp < ALIGN_MIN_RESPONSE).sum()),
        "gate": {"min_response": ALIGN_MIN_RESPONSE, "min_shift_px": ALIGN_MIN_SHIFT_PX},
        "gated_pairs": gated_stats(neg),
        "shaky_pairs_over_1px": {
            "n": int((mag > 1).sum()),
            "changed_raw_pct": float(neg[mag > 1].changed_raw.mean() * 100) if (mag > 1).any() else None,
            "changed_aligned_pct": float(neg[mag > 1].changed_aligned.mean() * 100) if (mag > 1).any() else None,
        },
    }
    out.update(fig_tiling(df, boxes))
    fig_normalisation(df)
    out["failure_examples"] = fig_differencing(df, shifts, boxes)
    fig_augmentation(df, boxes)
    write_result("p3_preprocess", out)


if __name__ == "__main__":
    main()
