"""Frame loading, tiling, box->tile labels, and frame alignment.

Geometry (all in "kept" pixels, i.e. after the 2x downscale done at download):
  * Every kept frame is 1536 px wide. Height is 1152 (4:3 cameras) or 1024 (3:2).
  * We standardise to 1536x1024 by cropping rows off the TOP (sky). Ignitions
    start on the ground, so the top of the frame is the least useful part
    (SmokeyNet crops the top of the frame for the same reason).
  * The 1536x1024 frame is cut into an 8 x 5 grid of 224x224 tiles that overlap
    slightly (~37 px horizontally, 24 px vertically) so a small plume on a tile
    border is fully inside at least one tile.
"""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
from PIL import Image

from .common import FRAMES, KEEP_WIDTH

STD_W, STD_H = 1536, 1024
TILE = 224
N_COLS, N_ROWS = 8, 5
N_TILES = N_COLS * N_ROWS
XS = np.round(np.linspace(0, STD_W - TILE, N_COLS)).astype(int)
YS = np.round(np.linspace(0, STD_H - TILE, N_ROWS)).astype(int)
TILE_XY = np.array([(x, y) for y in YS for x in XS])  # row-major, (N_TILES, 2)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_frame(sequence: str, file: str) -> np.ndarray:
    """Return a uint8 RGB array of shape (1024, 1536, 3)."""
    im = np.asarray(Image.open(FRAMES / sequence / file).convert("RGB"))
    return standardise(im)


def standardise(im: np.ndarray) -> np.ndarray:
    h, w = im.shape[:2]
    if w != STD_W:
        im = cv2.resize(im, (STD_W, round(h * STD_W / w)), interpolation=cv2.INTER_AREA)
        h = im.shape[0]
    if h >= STD_H:
        return im[h - STD_H:]  # crop the top (sky)
    pad = np.zeros((STD_H - h, STD_W, 3), dtype=im.dtype)  # rare: shorter frames
    return np.concatenate([pad, im], axis=0)


def crop_offset(orig_w: int, orig_h: int) -> int:
    """How many kept-pixel rows were cut from the top by standardise()."""
    kept_h = round(orig_h * KEEP_WIDTH / orig_w)
    return max(0, kept_h - STD_H)


def tiles(im: np.ndarray) -> np.ndarray:
    """(H, W, C) -> (N_TILES, TILE, TILE, C)."""
    return np.stack([im[y:y + TILE, x:x + TILE] for x, y in TILE_XY])


def box_to_tile_labels(box_orig, orig_w, orig_h, min_cover=0.3) -> np.ndarray:
    """Turn one smoke box (original-resolution pixels) into 40 tile labels.

    A tile is positive if it contains at least `min_cover` of the box area, or
    if the box covers at least `min_cover` of the tile. The first rule catches
    small early plumes; the second catches tiles fully inside a big plume.
    """
    s = KEEP_WIDTH / orig_w
    dy = crop_offset(orig_w, orig_h)
    x0, y0, x1, y1 = box_orig
    bx0, bx1 = x0 * s, x1 * s
    by0, by1 = y0 * s - dy, y1 * s - dy
    barea = max(1e-6, (bx1 - bx0) * (by1 - by0))
    lab = np.zeros(N_TILES, dtype=np.int8)
    for i, (tx, ty) in enumerate(TILE_XY):
        ix = max(0, min(bx1, tx + TILE) - max(bx0, tx))
        iy = max(0, min(by1, ty + TILE) - max(by0, ty))
        inter = ix * iy
        if inter / barea >= min_cover or inter / (TILE * TILE) >= min_cover:
            lab[i] = 1
    return lab


# ---------------------------------------------------------------- alignment
ALIGN_SCALE = 0.25  # estimate shifts on a 4x smaller grey image: fast and robust to noise
# Gate: only trust a shift when phase correlation is confident AND the shift is
# at least a pixel. Phase 3 found that low-confidence "shifts" are mostly clouds or
# lighting changes fooling the estimator, and that warping by sub-pixel amounts
# blurs the previous frame and ADDS spurious difference.
ALIGN_MIN_RESPONSE = 0.9
ALIGN_MIN_SHIFT_PX = 1.0


def gated_shift(dx: float, dy: float, response: float) -> tuple[float, float]:
    if response >= ALIGN_MIN_RESPONSE and (dx * dx + dy * dy) ** 0.5 >= ALIGN_MIN_SHIFT_PX:
        return dx, dy
    return 0.0, 0.0


def estimate_shift(prev: np.ndarray, cur: np.ndarray) -> tuple[float, float, float]:
    """Translation (dx, dy) that maps `prev` onto `cur`, via phase correlation.

    Phase correlation compares the two images in the frequency domain; a pure
    shift shows up as a single sharp peak whose position is the shift.
    Returns (dx, dy, response) with dx, dy in full standardised pixels.
    """
    a = _small_grey(prev)
    b = _small_grey(cur)
    win = _hann(a.shape)
    (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
    return dx / ALIGN_SCALE, dy / ALIGN_SCALE, resp


def apply_shift(im: np.ndarray, dx: float, dy: float) -> np.ndarray:
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(im, m, (im.shape[1], im.shape[0]), borderMode=cv2.BORDER_REPLICATE)


def _small_grey(im):
    g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, None, fx=ALIGN_SCALE, fy=ALIGN_SCALE, interpolation=cv2.INTER_AREA)
    return g.astype(np.float32)


@lru_cache(maxsize=4)
def _hann(shape):
    return cv2.createHanningWindow((shape[1], shape[0]), cv2.CV_32F)


def difference_image(cur: np.ndarray, prev: np.ndarray, gain: float = 2.0) -> np.ndarray:
    """Signed change encoded as an RGB uint8 image centred on grey 128.

    Pixels that got brighter (smoke over dark terrain) go above 128, pixels that
    got darker go below. `gain` amplifies faint changes so they are not lost
    when the image is quantised back to 8 bits.
    """
    d = cur.astype(np.int16) - prev.astype(np.int16)
    return np.clip(128 + gain * d, 0, 255).astype(np.uint8)


def normalise(batch_uint8: np.ndarray) -> np.ndarray:
    """(N, H, W, 3) uint8 -> (N, 3, H, W) float32 with ImageNet mean/std."""
    x = batch_uint8.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.transpose(0, 3, 1, 2).copy()
