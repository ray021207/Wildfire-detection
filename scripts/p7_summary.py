"""Phase 7b: count error categories.

Reads results/p7_error_labels.csv (one row per gallery item: id, category,
note), assigned by visually reviewing figures/p7_false_alarms_*.png and
figures/p7_misses_*.png, and writes results/p7_failures.json plus a bar chart.

Usage: python scripts/p7_summary.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import DATA, FIGURES, RESULTS, write_result  # noqa: E402


def main():
    items = pd.read_csv(DATA / "p7_items.csv")
    labels = pd.read_csv(RESULTS / "p7_error_labels.csv")
    d = items.merge(labels, on="id", how="left")
    assert d.category.notna().all(), "every gallery item needs a category"
    out = {}
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for j, kind in enumerate(("false_alarm", "miss")):
        s = d[d.kind == kind]
        c = s.category.value_counts()
        out[kind] = {"n_reviewed": int(len(s)), "counts": {k: int(v) for k, v in c.items()},
                     "night_items": int(s.night.sum()),
                     "by_split": {k: int(v) for k, v in s.split.value_counts().items()}}
        ax[j].barh(c.index[::-1], c.values[::-1], color="#C44E52" if kind == "false_alarm" else "#4C72B0")
        ax[j].set(title=f"{kind.replace('_', ' ')}s (n={len(s)})", xlabel="count")
    fig.tight_layout()
    fig.savefig(FIGURES / "p7_error_categories.png", dpi=110)
    plt.close(fig)
    write_result("p7_failures", out)


if __name__ == "__main__":
    main()
