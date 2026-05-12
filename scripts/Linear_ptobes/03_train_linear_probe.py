"""
Step 4a: Train and evaluate the linear probe.

Loads the Phase-3 activations from data/, fits a StandardScaler on the
training split only, trains a LogisticRegression(C=1.0, L2) on the
standardised training fingerprints, and evaluates on val and test.

Saves to models/ and results/:
  - models/linear_probe.pkl       trained sklearn classifier
  - models/scaler.pkl             fitted StandardScaler (fit on train only)
  - results/probe_direction.npy   raw weight vector (2304,) — the "sentiment
                                  direction" for later cosine-similarity
                                  comparison with the SAE
  - results/probe_metrics.json    accuracy / F1 / AUC / confusion matrix on
                                  train, val, test

Run from project root:
    source .venv/bin/activate
    python scripts/03_train_linear_probe.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score)
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

C_REG = 1.0
MAX_ITER = 1000
SEED = 42


def load_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    X = np.load(DATA_DIR / f"{split}_activations.npy")
    y = np.load(DATA_DIR / f"{split}_labels.npy")
    return X, y


def evaluate(name: str, clf, X, y) -> dict:
    pred = clf.predict(X)
    proba = clf.predict_proba(X)[:, 1]
    metrics = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "auc_roc": float(roc_auc_score(y, proba)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }
    print(f"  [{name:5s}] n={metrics['n']:4d}  "
          f"acc={metrics['accuracy']:.4f}  "
          f"f1={metrics['f1']:.4f}  "
          f"auc={metrics['auc_roc']:.4f}")
    print(f"           confusion = {metrics['confusion_matrix']}")
    return metrics


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    # --- 1. Load data ---
    print("[1/5] Loading activations...")
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")
    print(f"      train  X={X_train.shape}  y={y_train.shape}")
    print(f"      val    X={X_val.shape}    y={y_val.shape}")
    print(f"      test   X={X_test.shape}   y={y_test.shape}")

    # --- 2. Standardise (fit on train only — never on val/test) ---
    print("[2/5] Standardising features (fit on train, transform all)...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    print(f"      after scaling: train mean={X_train_s.mean():.4e}  "
          f"std={X_train_s.std():.4f}  (≈0, ≈1)")

    # --- 3. Train probe ---
    print(f"[3/5] Training LogisticRegression(C={C_REG}, "
          f"max_iter={MAX_ITER}, seed={SEED})...")
    clf = LogisticRegression(
        C=C_REG,
        max_iter=MAX_ITER,
        random_state=SEED,
        solver="lbfgs",   # default; explicit for clarity
        penalty="l2",     # default; explicit because the spec mandates L2
    )
    clf.fit(X_train_s, y_train)
    print(f"      converged in {clf.n_iter_[0]} iterations  "
          f"(limit={MAX_ITER})")

    # --- 4. Evaluate ---
    print("[4/5] Evaluating...")
    metrics = {
        "config": {"C": C_REG, "max_iter": MAX_ITER, "seed": SEED,
                   "penalty": "l2", "solver": "lbfgs"},
        "train": evaluate("train", clf, X_train_s, y_train),
        "val":   evaluate("val",   clf, X_val_s,   y_val),
        "test":  evaluate("test",  clf, X_test_s,  y_test),
    }

    # Quick overfit check
    gap = metrics["train"]["accuracy"] - metrics["test"]["accuracy"]
    print(f"      train-vs-test accuracy gap: {gap:+.4f}  "
          "(small gap = healthy generalisation)")

    # --- 5. Save artefacts ---
    print("[5/5] Saving artefacts...")
    with open(MODELS_DIR / "linear_probe.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open(MODELS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Raw weight vector — this IS the sentiment direction we will later
    # compare with the SAE's decoder vector via cosine similarity.
    direction = clf.coef_[0].astype(np.float32)   # (2304,)
    np.save(RESULTS_DIR / "probe_direction.npy", direction)

    with open(RESULTS_DIR / "probe_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"      saved models/linear_probe.pkl")
    print(f"      saved models/scaler.pkl")
    print(f"      saved results/probe_direction.npy   shape={direction.shape}  "
          f"||w||₂={np.linalg.norm(direction):.4f}")
    print(f"      saved results/probe_metrics.json")

    print(f"\nLinear probe trained. Test accuracy = "
          f"{metrics['test']['accuracy']:.4f}")
    if metrics["test"]["accuracy"] >= 0.85:
        print("Above the 85% stretch goal from the spec.")
    elif metrics["test"]["accuracy"] >= 0.75:
        print("Above the 75% MVP threshold from the spec.")
    else:
        print("Below the 75% MVP threshold — investigate before proceeding.")


if __name__ == "__main__":
    main()
