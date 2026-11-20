"""Phase 6: operational evaluation per fire sequence.

Alert logic: a frame alerts if its frame score (max tile score, or the
persistence score) is >= threshold. Per sequence:
  time-to-detection (TTD) = offset (minutes after first visible smoke) of the
                            first alert at/after t=0. No alert -> missed.
  false alert episode     = an alert that switches on in a smoke-free (t<0)
                            frame; consecutive alerting frames count once.
  false alerts per camera-day = total false episodes / smoke-free camera-hours * 24
     smoke-free camera-hours = sum over sequences of (#frames before t=0 * frame interval)

Two operating points per model, both fixed on VALIDATION and then applied to test:
  * "f1"     : threshold maximising validation frame F1
  * "fa_budget": lowest threshold whose validation false alerts/camera-day <= FA_BUDGET
And a full threshold sweep for the trade-off figure.

Writes results/p6_operational.json and figures/p6_*.png
Usage: python scripts/p6_operational.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import FIGURES, SEED, write_result  # noqa: E402
from src.evaluate import SCORES  # noqa: E402
from src.metrics import best_f1_threshold, simulate, summarise_operational  # noqa: E402

META = {"sequence", "file", "timestamp", "offset_s", "label", "split", "night"}
FA_BUDGET = 1.0
SPLITS = ("val", "test_seen", "test_unseen")
N_BOOT = 1000


def sweep(frames, col, thresholds):
    out = []
    for t in thresholds:
        s = summarise_operational(simulate(frames, col, t))
        out.append({"threshold": float(t), **s})
    return out


def boot_ci(per_seq, seed=SEED):
    rng = np.random.default_rng(seed)
    meds, fas, w10 = [], [], []
    n = len(per_seq)
    for _ in range(N_BOOT):
        b = per_seq.iloc[rng.integers(0, n, n)]
        ttd = np.where(b.detected, b.ttd_min, np.inf)
        meds.append(np.median(ttd))
        fas.append(b.false_episodes.sum() / b.neg_hours.sum() * 24)
        w10.append((b.ttd_min <= 10).mean() * 100)

    def ci(v):
        lo, hi = np.percentile(v, [2.5, 97.5])
        return [None if np.isinf(lo) else float(lo), None if np.isinf(hi) else float(hi)]

    return {"median_ttd_all_fires_ci95": ci(meds), "false_alerts_per_camera_day_ci95": ci(fas),
            "pct_within_10min_ci95": ci(w10)}


def main():
    frames = pd.read_csv(SCORES / "frame_scores.csv")
    models = [c for c in frames.columns if c not in META]
    by_split = {sp: frames[frames.split == sp] for sp in SPLITS}
    results, curves = {}, {}
    for m in models:
        val = by_split["val"]
        thr_grid = np.unique(np.quantile(frames[m], np.r_[np.linspace(0.3, 0.99, 60), 1 - np.logspace(-2, -5, 40)]))
        val_sweep = sweep(val, m, thr_grid)
        ok = [r for r in val_sweep if r["false_alerts_per_camera_day"] is not None
              and r["false_alerts_per_camera_day"] <= FA_BUDGET]
        ops = {"f1": best_f1_threshold(val.label.values, val[m].values)}
        if ok:
            ops["fa_budget"] = min(r["threshold"] for r in ok)
        results[m] = {}
        for op, thr in ops.items():
            results[m][op] = {"threshold": thr}
            for sp in SPLITS:
                ps = simulate(by_split[sp], m, thr)
                results[m][op][sp] = {**summarise_operational(ps), **(boot_ci(ps) if sp != "val" else {}),
                                      "ttd_per_sequence": dict(zip(ps.sequence, ps.ttd_min.round(2)))}
        curves[m] = {sp: sweep(by_split[sp], m, thr_grid) for sp in ("test_seen", "test_unseen")}
        print(m, {op: {sp: (round(results[m][op][sp]["median_ttd_min_detected_only"] or -1, 1),
                            round(results[m][op][sp]["false_alerts_per_camera_day"] or 0, 2)) for sp in SPLITS}
                  for op in ops}, flush=True)

    plot_tradeoff(curves, results)
    plot_ttd_hist(results)
    hours = {sp: float(simulate(by_split[sp], models[0], 2.0).neg_hours.sum()) for sp in SPLITS}
    write_result("p6_operational", {
        "fa_budget_per_camera_day": FA_BUDGET,
        "negative_camera_hours": hours,
        "n_sequences": {sp: int(by_split[sp].sequence.nunique()) for sp in SPLITS},
        "formula": "false_alerts_per_camera_day = sum(false_alert_episodes) / sum(n_frames_before_t0 * interval_s / 3600) * 24",
        "operating_points": results,
        "sweeps": curves,
    })


KEY = {"p4_A2_finetune": ("A2 single-frame fine-tune", "#4C72B0"),
       "p4_A1_frozen_lr": ("A1 single-frame frozen+LR", "#8FAADC"),
       "p5_B2_finetune_diff": ("B2 frame+diff fine-tune (from ImageNet)", "#C9C9C9"),
       "p5_B2w_finetune_diff_warm": ("B2-warm frame+diff fine-tune (from A2)", "#DD8452"),
       "p5_B1_frozen_lr_diff": ("B1 frame+diff frozen+LR", "#F2B48A")}


def label_of(m):
    base = m.split("+")[0]
    name, col = KEY.get(base, (base, "grey"))
    return (name + (" + persistence " + m.split("persist")[1] if "+persist" in m else "")), col, ("--" if "+persist" in m else "-")


def plot_tradeoff(curves, results):
    show = [m for m in curves if m.split("+")[0] in KEY]
    fig, ax = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    for j, sp in enumerate(("test_seen", "test_unseen")):
        for m in show:
            lab, col, ls = label_of(m)
            c = pd.DataFrame(curves[m][sp])
            c = c[c.false_alerts_per_camera_day > 0]
            x = c.false_alerts_per_camera_day
            ax[0, j].plot(x, c.median_ttd_min_all_fires, ls, color=col, label=lab, lw=1.6)
            ax[1, j].plot(x, c.pct_detected_within_10min, ls, color=col, label=lab, lw=1.6)
        ax[0, j].set(title=f"{sp.replace('_', ' ')} cameras: median time-to-detection (all fires)",
                     ylabel="median minutes after first visible smoke", xscale="log")
        ax[1, j].set(title=f"{sp.replace('_', ' ')} cameras: fires detected within 10 min",
                     ylabel="% of fires", xlabel="false alerts per camera-day (log)", xscale="log")
        for a in ax[:, j]:
            a.grid(alpha=0.3, which="both")
    ax[0, 0].legend(fontsize=8)
    fig.suptitle("Detection speed vs false-alarm rate as the alert threshold changes "
                 "(gaps = more than half of fires never detected)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "p6_tradeoff.png", dpi=110)
    plt.close(fig)


def plot_ttd_hist(results):
    show = [m for m in results if m.split("+")[0] in ("p4_A2_finetune", "p5_B2w_finetune_diff_warm")]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for j, sp in enumerate(("test_seen", "test_unseen")):
        for m in show:
            op = "fa_budget" if "fa_budget" in results[m] else "f1"
            v = np.array(list(results[m][op][sp]["ttd_per_sequence"].values()), dtype=float)
            v = np.sort(np.where(np.isnan(v), np.inf, v))
            lab, col, ls = label_of(m)
            ax[j].step(np.r_[0, v[np.isfinite(v)]], np.r_[0, np.arange(1, np.isfinite(v).sum() + 1) / len(v) * 100],
                       ls, color=col, where="post", label=f"{lab} [{op}]")
        ax[j].set(title=f"{sp.replace('_', ' ')}: cumulative % of fires detected", xlabel="minutes after first visible smoke",
                  xlim=(0, 40))
        ax[j].grid(alpha=0.3)
    ax[0].set_ylabel("% of fires detected")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "p6_ttd_cdf.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
