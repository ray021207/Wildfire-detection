"""Phase 8: export fine-tuned models to ONNX, measure CPU latency, try INT8.

For each model (A2 single-frame, B2-warm frame+diff):
  (B2-warm is used for B because B2 from ImageNet did not learn; see p5_temporal.)
  1. Export PyTorch -> ONNX (FP32), check outputs match PyTorch.
  2. Static INT8 quantisation (ONNX Runtime, QDQ format, per-channel weights),
     calibrated on 400 tiles from VALIDATION frames.
  3. CPU latency with ONNX Runtime:
       per tile  : batch of 1 tile, model only
       per frame : full pipeline = JPEG decode + crop + tile + normalise
                   (+ load previous frame, align, difference for B2) + 40 tiles
     median and p95 over repeated runs; frames/second = 1 / median frame latency.
  4. Accuracy of INT8 vs FP32 on the test frames: frame PR-AUC and the
     operational metrics at the same validation-chosen threshold.
  5. For reference only: PyTorch FP16 latency on the laptop GPU.
TensorRT is NOT tested (see NOTES.md).

Writes data/models/*.onnx, results/p8_edge.json.
Usage: python scripts/p8_export.py
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import MODELS, SEED, read_result, seed_everything, write_result  # noqa: E402
from src.data import frame_inputs, load_index  # noqa: E402
from src.evaluate import load_scores  # noqa: E402
from src.imaging import N_TILES, difference_image, normalise  # noqa: E402
from src.metrics import simulate, summarise_operational  # noqa: E402
from src.models import TileClassifier  # noqa: E402

CANDIDATES = {"p4_A2_finetune": False, "p5_B2w_finetune_diff_warm": True}
N_TILE_RUNS, N_FRAME_RUNS = 300, 60


def prep(row, temporal):
    cur, prev = frame_inputs(row, temporal)
    x = normalise(cur)
    if temporal:
        x = np.concatenate([x, normalise(np.stack([difference_image(c, p) for c, p in zip(cur, prev)]))], axis=1)
    return x


class Calib(CalibrationDataReader):
    def __init__(self, rows, temporal, name):
        tiles = [prep(r, temporal) for r in rows]
        x = np.concatenate(tiles)
        rng = np.random.default_rng(SEED)
        self.x = x[rng.choice(len(x), 400, replace=False)]
        self.name, self.i = name, 0

    def get_next(self):
        if self.i >= len(self.x):
            return None
        self.i += 8
        return {self.name: self.x[self.i - 8:self.i]}


def session(path, threads=None):
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def timeit(fn, n, warmup=5):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95)), "runs": n}


def run_frames(sess, df_rows, temporal, bs_frames=4):
    name = sess.get_inputs()[0].name
    out = np.zeros((len(df_rows), N_TILES), np.float32)
    for i in range(0, len(df_rows), bs_frames):
        x = np.concatenate([prep(r, temporal) for r in df_rows[i:i + bs_frames]])
        logit = sess.run(None, {name: x})[0].reshape(-1, N_TILES)
        out[i:i + bs_frames] = 1 / (1 + np.exp(-logit))
    return out


def main():
    seed_everything()
    df = load_index()
    rows = list(df.itertuples(index=False))
    results = {"cpu_threads_used": os.cpu_count(), "onnxruntime": ort.__version__,
               "tensorrt": "not tested: TensorRT is not installed and would add several GB, "
                           "which does not fit the 10 GB folder budget"}
    for name, temporal in CANDIDATES.items():
        if not (MODELS / f"{name}.pt").exists():
            continue
        in_ch = 6 if temporal else 3
        model = TileClassifier(in_ch).eval()
        model.load_state_dict(torch.load(MODELS / f"{name}.pt", map_location="cpu", weights_only=True))
        fp32 = MODELS / f"{name}.onnx"
        dummy = torch.randn(2, in_ch, 224, 224)
        torch.onnx.export(model, (dummy,), str(fp32), input_names=["tiles"], output_names=["logits"],
                          dynamic_axes={"tiles": {0: "n"}, "logits": {0: "n"}}, opset_version=17, dynamo=False)
        pre = MODELS / f"{name}.pre.onnx"
        quant_pre_process(str(fp32), str(pre))
        int8 = MODELS / f"{name}.int8.onnx"
        val_rows = [rows[i] for i in np.random.default_rng(SEED).choice(np.where(df.split == "val")[0], 20, replace=False)]
        quantize_static(str(pre), str(int8), Calib(val_rows, temporal, "tiles"), quant_format=QuantFormat.QDQ,
                        per_channel=True, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
        pre.unlink()

        s32, s8 = session(fp32), session(int8)
        x1 = prep(rows[len(rows) // 2], temporal)
        with torch.no_grad():
            ref = model(torch.from_numpy(x1)).numpy()
        max_abs_diff = float(np.abs(s32.run(None, {"tiles": x1})[0] - ref).max())

        r = {"onnx_fp32_mb": fp32.stat().st_size / 1e6, "onnx_int8_mb": int8.stat().st_size / 1e6,
             "onnx_vs_torch_max_abs_logit_diff": max_abs_diff}
        test_rows = [rows[i] for i in range(len(rows)) if rows[i].split.startswith("test")]
        for tag, sess in (("fp32", s32), ("int8", s8)):
            one = x1[:1]
            r[f"{tag}_per_tile"] = timeit(lambda: sess.run(None, {"tiles": one}), N_TILE_RUNS)
            pick = [test_rows[i] for i in np.random.default_rng(1).choice(len(test_rows), N_FRAME_RUNS)]
            it = iter(pick * 2)
            r[f"{tag}_per_frame_end_to_end"] = timeit(lambda: sess.run(None, {"tiles": prep(next(it), temporal)}),
                                                      N_FRAME_RUNS, warmup=3)
            r[f"{tag}_fps"] = 1000 / r[f"{tag}_per_frame_end_to_end"]["median_ms"]
            r[f"{tag}_per_frame_model_only"] = timeit(lambda: sess.run(None, {"tiles": x1}), 40, warmup=3)
        # accuracy: INT8 vs the PyTorch scores on all test frames
        print(f"{name}: scoring test frames with INT8", flush=True)
        test_idx = np.where(df.split.str.startswith("test"))[0]
        q8 = run_frames(s8, test_rows, temporal)
        q32 = run_frames(s32, test_rows, temporal)
        thr = read_result(name)["threshold_from_val"]
        fr = df.iloc[test_idx].copy()
        fr["fp32"], fr["int8"] = q32.max(1), q8.max(1)
        r["onnx_fp32_vs_training_scores_max_abs_diff"] = float(np.abs(load_scores(name)[test_idx] - q32).max())
        acc = {}
        for sp in ("test_seen", "test_unseen"):
            d = fr[fr.split == sp]
            acc[sp] = {}
            for tag in ("fp32", "int8"):
                acc[sp][tag] = {"frame_pr_auc": float(average_precision_score(d.label, d[tag])),
                                **{k: v for k, v in summarise_operational(simulate(d, tag, thr)).items()
                                   if k in ("median_ttd_min_detected_only", "pct_detected_within_10min",
                                            "false_alerts_per_camera_day", "pct_never_detected")}}
        r["accuracy_test"] = acc
        r["int8_vs_fp32_mean_abs_frame_score_diff"] = float(np.abs(fr.fp32 - fr.int8).mean())
        r["threshold"] = thr
        if torch.cuda.is_available():
            mg = model.cuda().half()
            xg = torch.from_numpy(x1).cuda().half()

            def g():
                with torch.no_grad():
                    mg(xg)
                torch.cuda.synchronize()
            r["reference_gpu_pytorch_fp16_40_tiles"] = {**timeit(g, 100), "device": torch.cuda.get_device_name(0)}
        results[name] = r
        print(name, {k: v for k, v in r.items() if "per_frame" in k or "mb" in k}, flush=True)
    write_result("p8_edge", results)


if __name__ == "__main__":
    main()
