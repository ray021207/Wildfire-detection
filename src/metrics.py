"""Classification metrics with sequence-level bootstrap CIs, and the
operational alert metrics (time-to-detection, false alerts per camera-day)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

N_BOOT = 1000


def prf(y, s, thr):
    p, r, f, _ = precision_recall_fscore_support(y, s >= thr, average="binary", zero_division=0)
    return {"precision": float(p), "recall": float(r), "f1": float(f)}


def best_f1_threshold(y, s):
    qs = np.unique(np.quantile(s, np.linspace(0.01, 0.995, 300)))
    f1s = [prf(y, s, t)["f1"] for t in qs]
    return float(qs[int(np.argmax(f1s))])


def metrics_with_ci(y, s, groups, thr, seed=0, n_boot=N_BOOT):
    """Precision/recall/F1 at `thr` and PR-AUC, with 95% CIs from a bootstrap that
    resamples whole SEQUENCES (frames inside a sequence are strongly correlated,
    so resampling single frames would make the intervals far too narrow)."""
    y, s, groups = np.asarray(y), np.asarray(s), np.asarray(groups)
    point = {**prf(y, s, thr), "pr_auc": float(average_precision_score(y, s)) if y.any() else None,
             "threshold": float(thr), "n": int(len(y)), "n_pos": int(y.sum()), "n_sequences": int(len(np.unique(groups)))}
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    pos = {g: np.where(groups == g)[0] for g in ug}
    boots = {k: [] for k in ("precision", "recall", "f1", "pr_auc")}
    for _ in range(n_boot):
        idx = np.concatenate([pos[g] for g in rng.choice(ug, len(ug), replace=True)])
        yy, ss = y[idx], s[idx]
        if yy.min() == yy.max():
            continue
        m = prf(yy, ss, thr)
        for k in ("precision", "recall", "f1"):
            boots[k].append(m[k])
        boots["pr_auc"].append(average_precision_score(yy, ss))
    for k, v in boots.items():
        if v:
            point[f"{k}_ci95"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return point


# --------------------------------------------------------------- operational
def persistence_score(frames: pd.DataFrame, score_col: str, k: int, n: int) -> np.ndarray:
    """K-of-N persistence turned into a score.

    "Alert if at least k of the last n frame scores exceed thr" is true exactly
    when the k-th largest of those n scores exceeds thr. So the k-th largest
    score in the trailing window is itself a frame score, and every threshold
    tool (PR curves, sweeps) works on it unchanged. Frames must be in time order
    within each sequence; returns an array aligned with frames.index.
    """
    out = pd.Series(np.nan, index=frames.index)
    for _, d in frames.sort_values("timestamp").groupby("sequence", sort=False):
        s = d[score_col].values
        vals = np.empty(len(s))
        for t in range(len(s)):
            w = np.sort(s[max(0, t - n + 1): t + 1])[::-1]
            vals[t] = w[k - 1] if len(w) >= k else 0.0  # not enough history yet: no alert
        out.loc[d.index] = vals
    return out.values


def persistence(alert: np.ndarray, k: int, n: int) -> np.ndarray:
    """Alert at frame t only if >= k of the last n frames (t included) were positive."""
    if k <= 1 and n <= 1:
        return alert
    c = np.convolve(alert.astype(int), np.ones(n, int), mode="full")[: len(alert)]
    return c >= k


def simulate(frames: pd.DataFrame, score_col: str, thr: float, k: int = 1, n: int = 1) -> pd.DataFrame:
    """Run the alert logic over every sequence in time order.

    frames needs: sequence, timestamp, offset_s, <score_col> (= max tile score of the frame).
    Returns one row per sequence: time-to-detection, false alert episodes, negative hours.
    """
    rows = []
    for seq, d in frames.sort_values("timestamp").groupby("sequence", sort=False):
        a = persistence(d[score_col].values >= thr, k, n)
        off = d["offset_s"].values
        ts = d["timestamp"].values
        neg = off < 0
        # a false alert EPISODE starts whenever the alert switches on in smoke-free
        # frames; a run of consecutive alerts is one page to a human, not many.
        starts = a & ~np.r_[False, a[:-1]]
        n_false_episodes = int((starts & neg).sum())
        interval = float(np.median(np.diff(ts))) if len(ts) > 1 else 60.0
        neg_hours = float(neg.sum() * interval / 3600)
        hit = np.where(a & ~neg)[0]
        ttd = float(off[hit[0]] / 60) if len(hit) else np.nan
        rows.append({"sequence": seq, "ttd_min": ttd, "detected": bool(len(hit)),
                     "false_episodes": n_false_episodes, "false_frames": int((a & neg).sum()),
                     "neg_frames": int(neg.sum()), "neg_hours": neg_hours,
                     "pos_window_min": float(off.max() / 60)})
    return pd.DataFrame(rows)


def summarise_operational(per_seq: pd.DataFrame) -> dict:
    """False alerts per camera-day = (false alert episodes) / (smoke-free camera-hours) * 24."""
    hours = per_seq["neg_hours"].sum()
    eps = per_seq["false_episodes"].sum()
    ttd = per_seq["ttd_min"]
    det = per_seq["detected"]
    # treating misses as "infinitely late" gives an honest median over ALL fires;
    # if half or more are missed that median does not exist (None).
    med_all = float(np.median(np.where(det, ttd, np.inf))) if len(per_seq) else np.inf
    return {
        "n_sequences": int(len(per_seq)),
        "median_ttd_min_detected_only": float(ttd[det].median()) if det.any() else None,
        "median_ttd_min_all_fires": None if np.isinf(med_all) else med_all,
        "pct_detected_within_5min": float((ttd <= 5).mean() * 100),
        "pct_detected_within_10min": float((ttd <= 10).mean() * 100),
        "pct_never_detected": float((~det).mean() * 100),
        "false_alert_episodes": int(eps),
        "negative_camera_hours": float(hours),
        "false_alerts_per_camera_day": float(eps / hours * 24) if hours > 0 else None,
        "false_alert_frames": int(per_seq["false_frames"].sum()),
        "negative_frames": int(per_seq["neg_frames"].sum()),
    }
