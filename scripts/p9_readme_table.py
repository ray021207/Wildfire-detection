"""Phase 9: fill the README results table from results/*.json (no hand-typed numbers).

Replaces the text between <!-- RESULTS-TABLE:START --> and <!-- RESULTS-TABLE:END -->
in README.md. Also writes results/p9_table.json with the same numbers.
Usage: python scripts/p9_readme_table.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, read_result, write_result  # noqa: E402

ROWS = [
    ("A2: single frame", "p4_A2_finetune", "p4_A2_finetune"),
    ("A2 + persistence", "p4_A2_finetune+persist", "p4_A2_finetune"),
    ("B2-warm: frame + difference", "p5_B2w_finetune_diff_warm", "p5_B2w_finetune_diff_warm"),
    ("B2-warm + persistence", "p5_B2w_finetune_diff_warm+persist", "p5_B2w_finetune_diff_warm"),
]
OP = "f1"  # threshold maximising validation frame F1 (the FA-budget point misses most fires; see NOTES)


def fmt(v, nd=1, suffix=""):
    return "n/a" if v is None else f"{v:.{nd}f}{suffix}"


def main():
    p5 = read_result("p5_temporal")["models"]
    p6 = read_result("p6_operational")
    p8 = read_result("p8_edge")
    ops = p6["operating_points"]
    lines = ["| Model | Cameras | Median minutes to detection | Fires detected within 10 min "
             "| False alerts per camera-day | Frame PR-AUC | CPU ms per frame (median, end to end) |",
             "|---|---|---|---|---|---|---|"]
    table = []
    for label, key, lat_key in ROWS:
        name = key if key in ops else next((m for m in ops if m.startswith(key)), None)
        if name is None:
            continue
        shown = label + (f" ({name.split('persist')[1].replace('of', ' of ')})" if "+persist" in name else "")
        lat = f"{p8[lat_key]['fp32_per_frame_end_to_end']['median_ms']:.0f}" if lat_key in p8 else "not measured"
        for sp, sp_label in (("test_seen", "seen"), ("test_unseen", "unseen")):
            r = ops[name][OP][sp]
            pr = p5[name][f"{sp}_frame"]["pr_auc"]
            row = {"model": shown, "variant": name, "operating_point": OP, "cameras": sp_label,
                   "median_ttd_min_all_fires": r["median_ttd_min_all_fires"],
                   "median_ttd_ci95": r["median_ttd_all_fires_ci95"],
                   "pct_within_10min": r["pct_detected_within_10min"],
                   "false_alerts_per_camera_day": r["false_alerts_per_camera_day"],
                   "false_alerts_ci95": r["false_alerts_per_camera_day_ci95"],
                   "frame_pr_auc": pr, "cpu_ms_per_frame_fp32": lat}
            table.append(row)
            med = fmt(r["median_ttd_min_all_fires"]) if r["median_ttd_min_all_fires"] is not None else "not reached"
            fa_ci = r["false_alerts_per_camera_day_ci95"]
            lines.append(f"| {shown} | {sp_label} ({r['n_sequences']} fires) | {med} "
                         f"| {fmt(r['pct_detected_within_10min'], 0, '%')} "
                         f"| {fmt(r['false_alerts_per_camera_day'], 1)} [{fmt(fa_ci[0], 1)}-{fmt(fa_ci[1], 1)}] "
                         f"| {pr:.3f} | {lat} |")
    md = "\n".join(lines)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    readme = re.sub(r"(<!-- RESULTS-TABLE:START -->).*?(<!-- RESULTS-TABLE:END -->)",
                    lambda m: f"{m.group(1)}\n{md}\n{m.group(2)}", readme, flags=re.S)
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    # Derived numbers quoted in README/NOTES, computed here so none is typed by hand.
    derived = {}
    for m in ("p4_A2_finetune", "p5_B2w_finetune_diff_warm"):
        if m not in p8:
            continue
        r = p8[m]
        derived[m] = {
            "int8_size_reduction_x": r["onnx_fp32_mb"] / r["onnx_int8_mb"],
            "int8_frame_ms_saved_end_to_end": r["fp32_per_frame_end_to_end"]["median_ms"]
            - r["int8_per_frame_end_to_end"]["median_ms"],
            "int8_frame_pr_auc_drop": {sp: r["accuracy_test"][sp]["fp32"]["frame_pr_auc"]
                                       - r["accuracy_test"][sp]["int8"]["frame_pr_auc"]
                                       for sp in r["accuracy_test"]},
            "model_share_of_end_to_end_fp32": r["fp32_per_frame_model_only"]["median_ms"]
            / r["fp32_per_frame_end_to_end"]["median_ms"],
        }
    bc = read_result("p4_backbone_choice")
    derived["backbone_cpu_slowdown_effnet_vs_mobilenet_x"] = (
        bc["efficientnet_b0"]["cpu_ms_per_40_tiles_median"] / bc["mobilenet_v3_large"]["cpu_ms_per_40_tiles_median"])
    from src.common import folder_size_bytes
    storage = {p: folder_size_bytes(ROOT / p) / 1e9 for p in (".venv", "data", "data/frames", "data/features",
                                                               "data/models", ".git")}
    storage["total"] = folder_size_bytes(ROOT) / 1e9
    storage["budget"] = 10.0
    write_result("p9_table", {"rows": table, "derived": derived, "storage_gb": storage})
    print(md)


if __name__ == "__main__":
    main()
