"""Training-time augmentation for tiles. Only changes that a fixed tower camera
could plausibly produce on its own:

  * horizontal flip      - smoke can drift left or right; the world has no
                           preferred left/right. (No vertical flip: smoke rises,
                           and the sky is always on top.)
  * brightness/contrast  - time of day, sun angle, haze and auto-exposure.
  * small translation    - wind shake of the mast moves the image a few pixels.
  * mild blur            - focus drift and atmospheric shimmer.

Deliberately NOT used: big rotations, hue shifts, cut-out. Smoke is greyish-
white/brown, and a hue shift can turn a cloud into something smoke-coloured.
"""
import cv2
import numpy as np


def augment_tile(t: np.ndarray, rng: np.random.Generator, prev: np.ndarray | None = None):
    """Augment a uint8 tile. If `prev` is given (temporal model), apply the SAME
    geometric/photometric change to both so their difference stays meaningful."""
    flip = rng.random() < 0.5
    alpha = rng.uniform(0.75, 1.25)      # contrast
    beta = rng.uniform(-25, 25)           # brightness
    dx, dy = rng.integers(-6, 7, size=2)  # pixels of mast shake
    blur = rng.random() < 0.2

    def f(x):
        x = x[:, ::-1] if flip else x
        x = np.clip(x.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        x = cv2.warpAffine(x, m, (x.shape[1], x.shape[0]), borderMode=cv2.BORDER_REFLECT)
        if blur:
            x = cv2.GaussianBlur(x, (3, 3), 0)
        return np.ascontiguousarray(x)

    return f(t) if prev is None else (f(t), f(prev))
