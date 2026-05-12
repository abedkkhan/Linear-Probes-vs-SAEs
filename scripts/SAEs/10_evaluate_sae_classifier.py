"""
Step 5e (Script 2): Load the cutoff chosen on val and apply it to the test
set. Report final SAE classifier metrics.

Reads:   data/SAEs/test_sae_features_max.npy
         data/SAEs/test_labels.npy
         results/sae_cutoff.json
Writes:  results/sae_classifier_metrics.json
"""

import json
import os
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score)

DATA_DIR = "data/SAEs"
OUT_DIR = "results"


def main() -> None:
    with open(os.path.join(OUT_DIR, "sae_cutoff.json")) as fp:
        cfg = json.load(fp)
    feature_id = cfg["feature_id"]
    pooling    = cfg["pooling"]
    cutoff     = cfg["cutoff"]

    print(f"Loaded cutoff = {cutoff:.4f}  (feature {feature_id}, pooling={pooling})")

    X = np.load(os.path.join(DATA_DIR, f"test_sae_features_{pooling}.npy"))
    y = np.load(os.path.join(DATA_DIR, "test_labels.npy"))
    f = X[:, feature_id]

    pred  = (f <= cutoff).astype(int)
    score = -f   # for AUC: higher score = more likely positive (label 1)

    acc = accuracy_score(y, pred)
    f1  = f1_score(y, pred)
    auc = roc_auc_score(y, score)
    cm  = confusion_matrix(y, pred).tolist()

    print(f"\nTest results:")
    print(f"  accuracy = {acc:.4f}")
    print(f"  f1       = {f1:.4f}")
    print(f"  auc      = {auc:.4f}")
    print(f"  confusion matrix [[TN, FP], [FN, TP]] = {cm}")

    out = {
        "feature_id": feature_id,
        "pooling": pooling,
        "cutoff": cutoff,
        "test": {
            "accuracy": acc,
            "f1": f1,
            "auc": auc,
            "confusion_matrix": cm,
        },
    }
    out_path = os.path.join(OUT_DIR, "sae_classifier_metrics.json")
    with open(out_path, "w") as fp:
        json.dump(out, fp, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()