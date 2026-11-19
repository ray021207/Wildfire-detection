"""Shared paths, seeding and small helpers used by every script."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"            # temporary: one downloaded .tgz at a time
FRAMES = DATA / "frames"      # downscaled frames we keep, one folder per sequence
FEATS = DATA / "features"     # cached tile embeddings
MODELS = DATA / "models"      # trained weights (gitignored because data/ is)
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

SEED = 1234

# FIgLib download locations (verified Nov 2026)
FIGLIB_BASE = "https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data"
FIGLIB_TAR = f"{FIGLIB_BASE}/Tar"
FIGLIB_BB_CSV = f"{FIGLIB_BASE}/Miscellaneous/Labels/HPWREN-BB/CSV"

# Image geometry used after download. Originals are 3072x2048 (or similar);
# we keep a 2x-downscaled copy to stay under the storage budget.
KEEP_WIDTH = 1536


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def write_result(name: str, payload: dict) -> Path:
    """Every number reported in the docs must come from a file written here."""
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    print(f"wrote {path}")
    return path


def read_result(name: str) -> dict:
    with open(RESULTS / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def folder_size_bytes(path: Path) -> int:
    total = 0
    for dirpath, _, files in os.walk(path):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, fn))
            except OSError:
                pass
    return total


def parse_sequence_name(seq: str) -> dict:
    """'20170708_Whittier_syp-n-mobo-m' -> date, fire, camera, site.

    A few names use '-' instead of '_' between date and fire
    (e.g. '20230128-Cardboard-Fire-om-w-mobo-c'); handle both.
    """
    date = seq[:8]
    rest = seq[9:]
    if "_" in rest:
        fire, camera = rest.rsplit("_", 1)
    else:
        # dash-separated: camera is the last token group starting at the site code.
        parts = rest.split("-")
        # camera ids look like site-dir-imager[-c|-m]; imager tokens below
        imagers = {"mobo", "iqeye", "axis", "hikvision"}
        idx = next((i for i, p in enumerate(parts) if p in imagers), len(parts) - 1)
        cam_start = max(0, idx - 2)
        fire = "-".join(parts[:cam_start])
        camera = "-".join(parts[cam_start:])
    site = camera.split("-")[0]
    return {"sequence": seq, "date": date, "fire": fire, "camera": camera,
            "site": site, "event": f"{date}_{fire}"}
