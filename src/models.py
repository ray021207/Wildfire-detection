"""Backbone + heads.

Backbone choice: MobileNetV3-Large (ImageNet-pretrained, torchvision).
  * ~5.5M parameters and ~0.22 GFLOPs at 224x224, versus ~0.39 GFLOPs for
    EfficientNet-B0 at similar ImageNet accuracy. Every frame is 40 tiles, so
    per-tile cost is multiplied by 40: the cheaper network matters on a CPU.
  * Built from depthwise-separable convolutions + hard-swish, which quantise
    to INT8 well and are well supported by ONNX Runtime.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import DATA

os.environ.setdefault("TORCH_HOME", str(DATA / "torch_cache"))  # keep weights inside the budgeted folder

EMB_DIM = 960


def mobilenet_features():
    from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

    m = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    return m.features  # (B, 960, 7, 7) for 224 input


class Embedder(nn.Module):
    """Frozen backbone -> 960-d average-pooled embedding per tile."""

    def __init__(self):
        super().__init__()
        self.features = mobilenet_features()

    def forward(self, x):
        return F.adaptive_avg_pool2d(self.features(x), 1).flatten(1)


class TileClassifier(nn.Module):
    """Scores each tile; in_ch=3 for single frame, 6 for frame + difference image."""

    def __init__(self, in_ch: int = 3):
        super().__init__()
        self.features = mobilenet_features()
        if in_ch != 3:
            old = self.features[0][0]
            new = nn.Conv2d(in_ch, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
            with torch.no_grad():
                new.weight[:, :3] = old.weight
                # diff channels start as a scaled copy of the RGB filters: they
                # respond to edges/blobs in the change image from step one.
                new.weight[:, 3:] = 0.5 * old.weight
            self.features[0][0] = new
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(EMB_DIM, 1))

    def forward(self, x):  # (B, C, 224, 224) -> (B,) logits
        return self.head(F.adaptive_avg_pool2d(self.features(x), 1).flatten(1)).squeeze(1)


def mil_loss(tile_logits, frame_labels, n_tiles: int, neg_tile_weight: float = 0.5):
    """Multiple-instance loss.

    We know whether a FRAME has smoke, but not WHICH tile. So:
      * frame logit = max over its tiles; BCE against the frame label.
        ("at least one tile has smoke" for positives, "no tile" for negatives)
      * in a negative frame every tile is a reliable negative, so we also apply
        BCE to every tile of negative frames.
    """
    t = tile_logits.view(-1, n_tiles)
    frame_logit = t.max(1).values
    loss = F.binary_cross_entropy_with_logits(frame_logit, frame_labels.float())
    neg = frame_labels == 0
    if neg.any():
        loss = loss + neg_tile_weight * F.binary_cross_entropy_with_logits(t[neg], torch.zeros_like(t[neg]))
    return loss
