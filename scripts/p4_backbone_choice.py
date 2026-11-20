"""Phase 4 (support): measure the two candidate backbones on this CPU.

For MobileNetV3-Large and EfficientNet-B0 (torchvision, ImageNet weights):
parameter count, multiply-accumulate operations per 224x224 tile (counted with
forward hooks on conv/linear layers), and CPU latency for one frame's worth of
tiles (batch of 40) in PyTorch. Run on an otherwise idle machine.
Writes results/p4_backbone_choice.json.
Usage: python scripts/p4_backbone_choice.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import seed_everything, write_result  # noqa: E402
import src.models  # noqa: E402,F401  (sets TORCH_HOME inside the project)


def count_macs(model, x):
    macs = []

    def hook(m, inp, out):
        if isinstance(m, nn.Conv2d):
            k = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
            macs.append(out.numel() * k)
        elif isinstance(m, nn.Linear):
            macs.append(m.in_features * m.out_features * out.shape[0])

    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    with torch.no_grad():
        model(x)
    for h in hs:
        h.remove()
    return sum(macs) / x.shape[0]


def main():
    seed_everything()
    from torchvision.models import (EfficientNet_B0_Weights, MobileNet_V3_Large_Weights, efficientnet_b0,
                                    mobilenet_v3_large)
    cands = {"mobilenet_v3_large": (mobilenet_v3_large, MobileNet_V3_Large_Weights.IMAGENET1K_V2),
             "efficientnet_b0": (efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1)}
    out = {"torch_threads": torch.get_num_threads()}
    x40 = torch.randn(40, 3, 224, 224)
    for name, (fn, w) in cands.items():
        m = fn(weights=w).eval()
        params = sum(p.numel() for p in m.parameters())
        macs = count_macs(m, x40[:1])
        with torch.no_grad():
            for _ in range(3):
                m(x40)
            ts = []
            for _ in range(15):
                t0 = time.perf_counter()
                m(x40)
                ts.append((time.perf_counter() - t0) * 1000)
        out[name] = {"params_millions": params / 1e6, "gmacs_per_tile": macs / 1e9,
                     "imagenet_top1_reported_by_torchvision": w.meta["_metrics"]["ImageNet-1K"]["acc@1"],
                     "cpu_ms_per_40_tiles_median": float(np.median(ts)),
                     "cpu_ms_per_40_tiles_p95": float(np.percentile(ts, 95))}
        print(name, out[name], flush=True)
    write_result("p4_backbone_choice", out)


if __name__ == "__main__":
    main()
