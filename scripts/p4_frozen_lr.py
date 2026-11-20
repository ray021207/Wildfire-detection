"""Frozen embeddings + logistic regression, trained with multiple-instance
pseudo-labels. Used for Baseline A (single frame) and Model B1 (frame + diff).

The problem: training frames only have a FRAME label (smoke after t=0), but the
classifier works on TILES and only 1-3 of the 40 tiles actually contain smoke.
Calling all 40 tiles of a smoke frame "smoke" would teach the model that sky
and hills are smoke. Instead:
  * every tile of a smoke-free frame is a reliable negative;
  * round 0: in each smoke frame, pick the TOP_K tiles that look most unlike
    the same tile position before t=0 (distance to the pre-smoke mean
    embedding). These are the pseudo-positive tiles. This uses the past only to
    MAKE LABELS; the trained model itself sees one frame at a time.
  * rounds 1..R: train the logistic regression, then re-pick the top-K tiles of
    each smoke frame by the model's own score, and retrain.

Usage: python scripts/p4_frozen_lr.py --name A1_frozen_lr --features cur
       python scripts/p4_frozen_lr.py --name B1_frozen_lr_diff --features cur diff_al
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import FEATS, SEED, seed_everything, write_result  # noqa: E402
from src.evaluate import evaluate, save_scores  # noqa: E402

TOP_K = 2
ROUNDS = 3
MAX_NEG_TILES = 60000


def load_features(kinds):
    arrs = [np.load(FEATS / f"{k}.npy") for k in kinds]
    return np.concatenate(arrs, axis=2)


def novelty_init(X, df, rows):
    """Distance of each tile from the mean pre-smoke embedding at the same position."""
    out = np.zeros((len(rows), X.shape[1]), np.float32)
    ref = {}
    for seq, d in df[df.label == 0].groupby("sequence"):
        ref[seq] = X[d.index.values].astype(np.float32).mean(0)
    for j, i in enumerate(rows):
        r = ref.get(df.sequence[i])
        out[j] = 0 if r is None else ((X[i].astype(np.float32) - r) ** 2).sum(1)
    return out


def fit_mil(X, df, train_rows, C, rng):
    tr = df.loc[train_rows]
    neg_rows = tr.index[tr.label == 0].values
    pos_rows = tr.index[tr.label == 1].values
    neg = X[neg_rows].reshape(-1, X.shape[2])
    neg = neg[rng.choice(len(neg), min(MAX_NEG_TILES, len(neg)), replace=False)].astype(np.float32)
    score = novelty_init(X, df, pos_rows)
    scaler, clf = None, None
    for _ in range(ROUNDS + 1):
        top = np.argsort(-score, axis=1)[:, :TOP_K]
        pos = np.concatenate([X[pos_rows[j], top[j]] for j in range(len(pos_rows))]).astype(np.float32)
        Xtr = np.concatenate([neg, pos])
        ytr = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
        scaler = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(scaler.transform(Xtr), ytr)
        score = predict(X[pos_rows], scaler, clf)
    return scaler, clf


def predict(Xf, scaler, clf):
    n, t, d = Xf.shape
    out = np.zeros(n * t, np.float32)
    flat = Xf.reshape(-1, d)
    for s in range(0, len(flat), 100000):
        out[s:s + 100000] = clf.predict_proba(scaler.transform(flat[s:s + 100000].astype(np.float32)))[:, 1]
    return out.reshape(n, t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--features", nargs="+", required=True)
    args = ap.parse_args()
    seed_everything()
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(FEATS / "frames.csv")
    X = load_features(args.features)
    train_rows = df.index[df.split == "train"].values
    val = (df.split == "val").values
    search = {}
    best = None
    for C in (0.01, 0.1, 1.0):
        scaler, clf = fit_mil(X, df, train_rows, C, rng)
        sv = predict(X[val], scaler, clf).max(1)
        ap_val = float(average_precision_score(df.label[val], sv))
        search[str(C)] = ap_val
        print(f"C={C}: val frame PR-AUC {ap_val:.3f}", flush=True)
        if best is None or ap_val > best[0]:
            best = (ap_val, C, scaler, clf)
    _, C, scaler, clf = best
    scores = predict(X, scaler, clf)
    save_scores(args.name, scores)
    res = evaluate(df, scores)
    write_result(args.name, {"model": args.name, "features": args.features, "top_k": TOP_K,
                             "mil_rounds": ROUNDS, "C_search_val_pr_auc": search, "C": C, **res})


if __name__ == "__main__":
    main()
