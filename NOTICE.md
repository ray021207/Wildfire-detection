# Notices and third-party licenses

## 1. Code: MIT

The source code in `src/` and `scripts/` is released under the MIT License (see `LICENSE`),
Copyright (c) 2026 Reshma Panibhate.

## 2. Data: HPWREN FIgLib, CC BY-NC 4.0

This project uses the HPWREN Fire Ignition images Library (FIgLib).

> Data courtesy of HPWREN, University of California San Diego (https://www.hpwren.ucsd.edu/).

- Archived HPWREN data, including FIgLib, is copyright © The Regents of the University of
  California and is licensed under the **Creative Commons Attribution-NonCommercial 4.0
  International License (CC BY-NC 4.0)**: https://creativecommons.org/licenses/by-nc/4.0/
- Terms: https://www.hpwren.ucsd.edu/cc.html. Dataset page: https://www.hpwren.ucsd.edu/FIgLib/
- Please also cite: Dewangan, A., et al. (2022). FIgLib & SmokeyNet: Dataset and Deep Learning Model
  for Real-Time Wildland Fire Smoke Detection. *Remote Sensing*, 14(4), 1007.

**What this repository contains from the data, and under which license:**

| Path | Content | License |
|---|---|---|
| `figures/` | Plots, some showing FIgLib frames that were downscaled, cropped, tiled, differenced and annotated | CC BY-NC 4.0, with the attribution above. Changes were made (as listed). |
| `results/` | Numbers computed from FIgLib (metrics, sequence names, frame counts, manual error labels) | CC BY-NC 4.0, with the attribution above |
| raw images | **Not included.** The scripts download them from HPWREN; `data/` is excluded from the repository. | n/a |

Because of the NonCommercial term, the figures and data-derived results may not be used for
commercial purposes under this license. HPWREN offers a separate no-fee Commercial Use Permission
on request (see the terms page above). This project is a non-commercial research/portfolio project.

HPWREN data is provided "as is", without warranty.

## 3. Bounding-box labels

The 9 bounding-box CSV files used for tile-level evaluation are part of FIgLib (HPWREN-BB labels)
and fall under the same CC BY-NC 4.0 terms. They are downloaded by `scripts/p1_download.py` and
are not redistributed here.

## 4. Pretrained model weights

The MobileNetV3-Large (IMAGENET1K_V2) and EfficientNet-B0 (IMAGENET1K_V1) ImageNet weights are
downloaded at run time through torchvision; they are **not** redistributed in this repository.
torchvision is licensed under the BSD 3-Clause License
(https://github.com/pytorch/vision/blob/main/LICENSE). Pretrained weights may carry the terms of
their training data (ImageNet); see the torchvision documentation for each model.

The fine-tuned weights and ONNX files produced by this project are kept in `data/` and are not
published.

## 5. Python dependencies

The packages in `requirements.txt` (PyTorch, torchvision, NumPy, pandas, scikit-learn, OpenCV,
Pillow, matplotlib, ONNX, ONNX Runtime, and others) are installed by the user and are not
redistributed here. Each is used under its own open-source license (BSD, MIT, Apache-2.0 or
similar, as published by each project).
