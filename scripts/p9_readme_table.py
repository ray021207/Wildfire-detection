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
    ("A1 single frame, frozen + LR", "p4_A1_frozen_lr", None),
    ("A2 single frame, fine-tuned", "p4_A2_finetune", "p4_A2_finetune"),
    ("A2 + persistence", "p4_A2_finetune+persist", "p4_A2_finetune"),
    ("B1 frame + diff, frozen + LR", "p5_B1_frozen_lr_diff", None),
    ("B2 frame + diff, fine-tuned from ImageNet", "p5_B2_finetune_diff", None),
    ("B2-warm frame + diff, fine-tuned from A2", "p5_B2w_finetune_diff_warm", "p5_B2w_finetune_diff_warm"),
    ("B2-warm + persistence", "p5_B2w_finetune_diff_warm+persist", "p5_B2w_finetune_diff_warm"),
]


def fmt(v, nd=1, suffix=""):
    return "n/a" if v is None else f"{v:.{nd}f}{suffix}"


def main():
    p6 = read_result("p6_operational")
    p8 = read_result("p8_edge")
    ops = p6["operating_points"]
    lines = ["| Model | Cameras | Median time-to-detection (min, all fires) | % fires detected within 10 min "
             "| False alerts per camera-day | CPU latency per frame, ms (FP32 / INT8) |",
             "|---|---|---|---|---|---|"]
    table = []
    for label, key, lat_key in ROWS:
        name = key if key in ops else next((m for m in ops if m.startswith(key)), None)
        if name is None:
            continue
        op = "fa_budget" if "fa_budget" in ops[name] else "f1"
        lat = "not measured"
        if lat_key and lat_key in p8:
            lat = (f"{p8[lat_key]['fp32_per_frame_end_to_end']['median_ms']:.0f} / "
                   f"{p8[lat_key]['int8_per_frame_end_to_end']['median_ms']:.0f}")
        for sp, sp_label in (("test_seen", "seen"), ("test_unseen", "unseen")):
            r = ops[name][op][sp]
            row = {"model": label, "variant": name, "operating_point": op, "cameras": sp_label,
                   "median_ttd_min_all_fires": r["median_ttd_min_all_fires"],
                   "pct_within_10min": r["pct_detected_within_10min"],
                   "false_alerts_per_camera_day": r["false_alerts_per_camera_day"], "latency": lat}
            table.append(row)
            med = fmt(r["median_ttd_min_all_fires"]) if r["median_ttd_min_all_fires"] is not None else "not reached (>50% missed)"
            lines.append(f"| {label} | {sp_label} ({r['n_sequences']} fires) | {med} | {fmt(r['pct_detected_within_10min'], 0, '%')} "
                         f"| {fmt(r['false_alerts_per_camera_day'], 2)} | {lat} |")
    md = "\n".join(lines)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    readme = re.sub(r"(<!-- RESULTS-TABLE:START -->).*?(<!-- RESULTS-TABLE:END -->)",
                    lambda m: f"{m.group(1)}\n{md}\n{m.group(2)}", readme, flags=re.S)
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    write_result("p9_table", {"rows": table})
    print(md)


if __name__ == "__main__":
    main()
