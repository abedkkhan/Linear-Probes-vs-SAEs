"""
Step 5f: Train small logistic regressions on the top-K SAE features (by
absolute training correlation with the label). Sweep K to see how accuracy
scales as we let the classifier use more SAE features.

This tests whether sentiment is "distributed" across many SAE features.
- K=1   -> single-feature classifier (similar to step 5e)
- K=10  -> small interpretable feature set
- K=100 -> medium
- All   -> roughly equivalent to a full probe on the SAE basis

Reads:   data/SAEs/{train,val,test}_sae_features_max.npy
         data/SAEs/{train,val,test}_labels.npy
         results/train_sae_feature_correlations_max.npy   (16384,)
Writes:  results/sae_topk_metrics.json
"""

import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score)
from sklearn.preprocessing import StandardScaler

DATA_DIR = "data/SAEs"
OUT_DIR = "results"
POOLING = "max"
K_VALUES = [1, 5, 10, 20, 50, 100, 500, 1000]


def evaluate(clf, scaler, X, y):
    Xs = scaler.transform(X)
    pred = clf.predict(Xs)
    proba = clf.predict_proba(Xs)[:, 1]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "auc": float(roc_auc_score(y, proba)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load full feature matrices and labels for all splits.
    Xtr = np.load(os.path.join(DATA_DIR, f"train_sae_features_{POOLING}.npy"))
    ytr = np.load(os.path.join(DATA_DIR, "train_labels.npy"))
    Xva = np.load(os.path.join(DATA_DIR, f"val_sae_features_{POOLING}.npy"))
    yva = np.load(os.path.join(DATA_DIR, "val_labels.npy"))
    Xte = np.load(os.path.join(DATA_DIR, f"test_sae_features_{POOLING}.npy"))
    yte = np.load(os.path.join(DATA_DIR, "test_labels.npy"))

    # Load per-feature correlations computed in step 5d
    corrs = np.load(os.path.join(OUT_DIR, f"train_sae_feature_correlations_{POOLING}.npy"))
    # Rank features by |corr| (descending). Indices into the 16384 axis.
    feature_ranking = np.argsort(-np.abs(corrs))

    print(f"Train: {Xtr.shape}  Val: {Xva.shape}  Test: {Xte.shape}")
    print(f"Top-10 features by |corr|: {feature_ranking[:10].tolist()}")
    print(f"|corr| of top feature:     {abs(corrs[feature_ranking[0]]):.4f}")
    print(f"|corr| of 10th feature:    {abs(corrs[feature_ranking[9]]):.4f}\n")

    results = {}

    for K in K_VALUES:
        if K > corrs.size:
            continue
        idx = feature_ranking[:K]                          # the top-K feature columns
        Xtr_k = Xtr[:, idx]
        Xva_k = Xva[:, idx]
        Xte_k = Xte[:, idx]

        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr_k)

        clf = LogisticRegression(
            C=1.0,
            max_iter=2000,
            random_state=42,
            penalty="l2",
            solver="lbfgs",
        )
        clf.fit(Xtr_s, ytr)

        train_acc = accuracy_score(ytr, clf.predict(Xtr_s))
        val_m  = evaluate(clf, scaler, Xva_k, yva)
        test_m = evaluate(clf, scaler, Xte_k, yte)

        results[str(K)] = {
            "feature_ids_used": idx.tolist(),
            "train_accuracy": float(train_acc),
            "val":  val_m,
            "test": test_m,
        }

        print(f"K={K:>5}  train acc={train_acc:.4f}  "
              f"val acc={val_m['accuracy']:.4f}  "
              f"test acc={test_m['accuracy']:.4f}  "
              f"test f1={test_m['f1']:.4f}  test auc={test_m['auc']:.4f}")

    # Save
    out_path = os.path.join(OUT_DIR, "sae_topk_metrics.json")
    with open(out_path, "w") as fp:
        json.dump({"pooling": POOLING, "K_values": K_VALUES, "results": results}, fp, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()