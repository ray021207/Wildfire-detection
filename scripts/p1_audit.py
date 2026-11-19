"""Phase 1b: audit the downloaded sequences.

Builds data/frame_index.csv (one row per kept frame) and writes
results/p1_audit.json plus figures/p1_*.png.

Usage: python scripts/p1_audit.py
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import DATA, FIGURES, FRAMES, parse_sequence_name, write_result  # noqa: E402

PACIFIC = ZoneInfo("America/Los_Angeles")
# Mean grey level (0-255) of a 64x48 thumbnail below which we call a frame "night".
# Chosen by eye on the brightness histogram (figures/p1_brightness.png): there is a
# clear gap between dark night frames and daytime frames.
NIGHT_BRIGHTNESS = 50


def load_boxes() -> pd.DataFrame:
    rows = []
    for csv in sorted((DATA / "labels").glob("*.csv")):
        df = pd.read_csv(csv)
        for _, r in df.iterrows():
            fn = Path(str(r["Filename"])).name
            rows.append({"sequence": csv.stem, "file": fn, "x0": r["MinX"], "y0": r["MinY"],
                         "x1": r["MaxX"], "y1": r["MaxY"]})
    return pd.DataFrame(rows)


def build_index() -> pd.DataFrame:
    rows = []
    for seq_dir in sorted(p for p in FRAMES.iterdir() if (p / "meta.json").exists()):
        meta = json.loads((seq_dir / "meta.json").read_text())
        info = parse_sequence_name(seq_dir.name)
        orig = max(meta["original_sizes"], key=meta["original_sizes"].get)
        ow, oh = map(int, orig.split("x"))
        for f in sorted(seq_dir.glob("*.jpg")):
            ts, off = f.stem.split("_")
            thumb = np.asarray(Image.open(f).convert("L").resize((64, 48)), dtype=np.float32)
            im_rgb = np.asarray(Image.open(f).convert("RGB").resize((64, 48)), dtype=np.float32)
            colourfulness = float(np.abs(im_rgb[..., 0] - im_rgb[..., 2]).mean())
            t = datetime.fromtimestamp(int(ts), tz=timezone.utc).astimezone(PACIFIC)
            rows.append({**info, "file": f.name, "timestamp": int(ts), "offset_s": int(off),
                         "orig_w": ow, "orig_h": oh, "brightness": float(thumb.mean()),
                         "colourfulness": colourfulness, "local_hour": t.hour + t.minute / 60})
    df = pd.DataFrame(rows)
    df["night"] = df["brightness"] < NIGHT_BRIGHTNESS
    return df


def main():
    FIGURES.mkdir(exist_ok=True)
    df = build_index()
    boxes = load_boxes()
    kept_seqs = set(df["sequence"])
    boxes = boxes[boxes["sequence"].isin(kept_seqs)]
    df = df.merge(boxes[["sequence", "file"]].drop_duplicates().assign(has_box=True),
                  on=["sequence", "file"], how="left")
    df["has_box"] = df["has_box"].fillna(False).astype(bool)
    df.to_csv(DATA / "frame_index.csv", index=False)
    boxes.to_csv(DATA / "boxes.csv", index=False)

    g = df.sort_values("timestamp").groupby("sequence")
    per_seq = pd.DataFrame({
        "n_frames": g.size(),
        "interval_s": g["timestamp"].apply(lambda s: float(np.median(np.diff(s.values))) if len(s) > 1 else np.nan),
        "min_offset_s": g["offset_s"].min(),
        "max_offset_s": g["offset_s"].max(),
        "night_share": g["night"].mean(),
        "site": g["site"].first(),
        "camera": g["camera"].first(),
        "event": g["event"].first(),
        "res": g.apply(lambda d: f"{d.orig_w.iloc[0]}x{d.orig_h.iloc[0]}", include_groups=False),
        "n_box_frames": g["has_box"].sum(),
    })
    per_seq.to_csv(DATA / "sequence_index.csv")

    # ---------- figures ----------
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax[0, 0].hist(per_seq["n_frames"], bins=30, color="#4C72B0")
    ax[0, 0].set(title="Frames per sequence", xlabel="frames", ylabel="sequences")
    ax[0, 1].hist(per_seq["interval_s"].dropna(), bins=30, color="#4C72B0")
    ax[0, 1].set(title="Median frame interval", xlabel="seconds", ylabel="sequences")
    res_counts = per_seq["res"].value_counts()
    ax[0, 2].barh(res_counts.index, res_counts.values, color="#4C72B0")
    ax[0, 2].set(title="Original resolution", xlabel="sequences")
    site_counts = per_seq["site"].value_counts()
    ax[1, 0].bar(site_counts.index, site_counts.values, color="#4C72B0")
    ax[1, 0].set(title="Sequences per camera site", ylabel="sequences")
    ax[1, 0].tick_params(axis="x", rotation=90, labelsize=7)
    ax[1, 1].hist(df["brightness"], bins=60, color="#4C72B0")
    ax[1, 1].axvline(NIGHT_BRIGHTNESS, color="crimson", ls="--", label="night threshold")
    ax[1, 1].set(title="Frame brightness (all frames)", xlabel="mean grey level", ylabel="frames")
    ax[1, 1].legend()
    ax[1, 2].hist(df["offset_s"] / 60, bins=80, color="#4C72B0")
    ax[1, 2].axvline(0, color="crimson", ls="--", label="first visible smoke (t=0)")
    ax[1, 2].set(title="Frame offsets relative to t=0", xlabel="minutes", ylabel="frames")
    ax[1, 2].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "p1_audit_overview.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["brightness"], bins=80, color="#4C72B0")
    ax.axvline(NIGHT_BRIGHTNESS, color="crimson", ls="--")
    ax.set(title="Brightness histogram used for the day/night split", xlabel="mean grey level", ylabel="frames")
    fig.tight_layout()
    fig.savefig(FIGURES / "p1_brightness.png", dpi=110)
    plt.close(fig)

    filmstrip(df, boxes)

    write_result("p1_audit", {
        "n_sequences": int(len(per_seq)),
        "n_fire_events": int(per_seq["event"].nunique()),
        "n_fire_days": int(df["date"].nunique()),
        "n_cameras": int(per_seq["camera"].nunique()),
        "n_sites": int(per_seq["site"].nunique()),
        "n_frames": int(len(df)),
        "frames_per_sequence": describe(per_seq["n_frames"]),
        "frame_interval_s": describe(per_seq["interval_s"].dropna()),
        "interval_s_value_counts": {str(k): int(v) for k, v in per_seq["interval_s"].round().value_counts().items()},
        "original_resolutions": {k: int(v) for k, v in res_counts.items()},
        "offset_range_min": [float(df["offset_s"].min() / 60), float(df["offset_s"].max() / 60)],
        "n_frames_before_t0": int((df["offset_s"] < 0).sum()),
        "n_frames_at_or_after_t0": int((df["offset_s"] >= 0).sum()),
        "night_brightness_threshold": NIGHT_BRIGHTNESS,
        "night_frame_share": float(df["night"].mean()),
        "sequences_mostly_night": int((per_seq["night_share"] > 0.5).sum()),
        "n_sequences_with_boxes": int((per_seq["n_box_frames"] > 0).sum()),
        "n_frames_with_boxes": int(df["has_box"].sum()),
        "box_frames_before_t0": int((df["has_box"] & (df["offset_s"] < 0)).sum()),
        "sequences_per_site": {k: int(v) for k, v in site_counts.items()},
    })


def describe(s):
    return {"min": float(s.min()), "median": float(s.median()), "mean": float(s.mean()), "max": float(s.max())}


def filmstrip(df, boxes):
    """One fire from 20 min before t=0 to 30 min after, with a zoom on the smoke."""
    seq = boxes["sequence"].value_counts().index[0] if len(boxes) else df["sequence"].iloc[0]
    d = df[df["sequence"] == seq].sort_values("offset_s")
    targets = [-20, -5, 0, 3, 6, 10, 20, 30]
    picks = [d.iloc[(d["offset_s"] / 60 - t).abs().argmin()] for t in targets]
    b = boxes[boxes["sequence"] == seq]
    # zoom window from the last box in the sequence (the largest plume), in kept-image coordinates
    zoom = None
    if len(b):
        last = b.sort_values("file").iloc[-1]
        scale = 1536 / picks[0]["orig_w"]
        cx, cy = (last.x0 + last.x1) / 2 * scale, (last.y0 + last.y1) / 2 * scale
        kept_h = round(picks[0]["orig_h"] * scale)
        x0 = int(np.clip(cx - 200, 0, 1536 - 400))
        y0 = int(np.clip(cy - 130, 0, kept_h - 260))
        zoom = (x0, y0, x0 + 400, y0 + 260)
    fig, ax = plt.subplots(2, len(picks), figsize=(2.6 * len(picks), 4.4))
    for i, r in enumerate(picks):
        im = Image.open(FRAMES / seq / r["file"]).convert("RGB")
        full = im.copy()
        if zoom:
            ImageDraw.Draw(full).rectangle(zoom, outline=(255, 40, 40), width=6)
        ax[0, i].imshow(full)
        ax[0, i].set_title(f"{r['offset_s'] / 60:+.0f} min", fontsize=10)
        ax[1, i].imshow(im.crop(zoom) if zoom else im)
        for a in ax[:, i]:
            a.axis("off")
    fig.suptitle(f"{seq}: full frame (top) and zoom on the smoke region (bottom). "
                 "t=0 = first visible smoke per FIgLib.", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "p1_filmstrip.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
