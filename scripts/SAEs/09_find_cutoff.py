"""
Step 5e (Script 1): Use the validation set to pick the best cutoff for the
top SAE feature (#14733). Save the cutoff to disk.

Reads:   data/SAEs/val_sae_features_max.npy
         data/SAEs/val_labels.npy
Writes:  results/sae_cutoff.json
"""

import json
import os
import numpy as np
from sklearn.metrics import accuracy_score

DATA_DIR = "data/SAEs"
OUT_DIR = "results"
FEATURE_ID = 14733
POOLING = "max"


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    X = np.load(os.path.join(DATA_DIR, f"val_sae_features_{POOLING}.npy"))
    y = np.load(os.path.join(DATA_DIR, "val_labels.npy"))
    f = X[:, FEATURE_ID]

    print(f"Val feature {FEATURE_ID}:  min={f.min():.3f}  max={f.max():.3f}  "
          f"mean={f.mean():.3f}  nonzero={(f!=0).mean()*100:.1f}%")

    # Try many candidate cutoffs (use the actual distribution of feature values)
    candidates = np.unique(np.concatenate([[0.0], np.quantile(f, np.linspace(0, 1, 200))]))

    best_acc, best_thr = -1.0, None
    for thr in candidates:
        pred = (f <= thr).astype(int)   # above cutoff -> negative (0), below -> positive (1)
        acc = accuracy_score(y, pred)
        if acc > best_acc:
            best_acc, best_thr = acc, thr

    print(f"\nBest cutoff = {best_thr:.4f}  (val accuracy = {best_acc:.4f})")

    out = {
        "feature_id": FEATURE_ID,
        "pooling": POOLING,
        "cutoff": float(best_thr),
        "val_accuracy_at_cutoff": float(best_acc),
        "rule": "if feature_value > cutoff -> predict negative (0), else positive (1)",
    }
    out_path = os.path.join(OUT_DIR, "sae_cutoff.json")
    with open(out_path, "w") as fp:
        json.dump(out, fp, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()