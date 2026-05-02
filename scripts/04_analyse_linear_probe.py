"""
Step 4d: Stand-alone analysis of the trained linear probe.

Loads the probe + scaler + activations + sentences, then produces:

  Tables (printed and saved):
    - results/probe_metrics_table.txt      (clean per-split metrics)
    - results/top_weight_dimensions.json   (top 20 +ve and -ve weight indices)
    - results/probe_qualitative.json       (best/worst classified sentences)

  Plots (PNGs in results/plots/):
    - confusion_val.png, confusion_test.png   confusion-matrix heatmaps
    - roc_curve_test.png                      ROC curve on test
    - score_distribution_test.png             logit histogram, pos vs neg
    - weight_magnitudes.png                   |w_i| histogram across 2304 dims

Run from project root:
    source .venv/bin/activate
    python scripts/04_analyse_linear_probe.py
"""

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score, roc_curve)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

sns.set_theme(style="whitegrid", context="talk")


def load_split(split: str):
    X = np.load(DATA_DIR / f"{split}_activations.npy")
    y = np.load(DATA_DIR / f"{split}_labels.npy")
    rows = json.loads((DATA_DIR / f"{split}_samples.json").read_text())
    return X, y, rows


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load everything ---
    with open(MODELS_DIR / "linear_probe.pkl", "rb") as f:
        clf = pickle.load(f)
    with open(MODELS_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    w = clf.coef_[0]                  # (2304,)
    b = float(clf.intercept_[0])

    X_train, y_train, _ = load_split("train")
    X_val, y_val, _ = load_split("val")
    X_test, y_test, test_rows = load_split("test")
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # ============================================================
    # 1. METRICS TABLE
    # ============================================================
    def metrics(name, X, y):
        pred = clf.predict(X)
        proba = clf.predict_proba(X)[:, 1]
        return {
            "split": name,
            "n": len(y),
            "accuracy": accuracy_score(y, pred),
            "f1": f1_score(y, pred),
            "auc": roc_auc_score(y, proba),
            "confusion": confusion_matrix(y, pred),
        }

    rows = [metrics("train", X_train_s, y_train),
            metrics("val",   X_val_s,   y_val),
            metrics("test",  X_test_s,  y_test)]

    table_lines = []
    table_lines.append(f"{'split':<6} {'n':>5} {'acc':>8} {'f1':>8} {'auc':>8}  confusion [[TN,FP],[FN,TP]]")
    table_lines.append("-" * 78)
    for r in rows:
        table_lines.append(
            f"{r['split']:<6} {r['n']:>5} "
            f"{r['accuracy']:>8.4f} {r['f1']:>8.4f} {r['auc']:>8.4f}  "
            f"{r['confusion'].tolist()}"
        )
    table_str = "\n".join(table_lines)
    print("\n=== METRICS ===")
    print(table_str)
    (RESULTS_DIR / "probe_metrics_table.txt").write_text(table_str + "\n")

    # ============================================================
    # 2. CONFUSION-MATRIX HEATMAPS (val + test)
    # ============================================================
    for r in rows:
        if r["split"] == "train":
            continue
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(r["confusion"], annot=True, fmt="d", cmap="Blues",
                    xticklabels=["pred neg", "pred pos"],
                    yticklabels=["true neg", "true pos"],
                    cbar=False, ax=ax)
        ax.set_title(f"Confusion matrix — {r['split']} "
                     f"(acc={r['accuracy']:.3f})")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"confusion_{r['split']}.png", dpi=150)
        plt.close(fig)
    print(f"\nSaved confusion_val.png, confusion_test.png")

    # ============================================================
    # 3. ROC CURVE (test)
    # ============================================================
    proba_test = clf.predict_proba(X_test_s)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba_test)
    auc_test = rows[2]["auc"]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"linear probe (AUC = {auc_test:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve — test set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "roc_curve_test.png", dpi=150)
    plt.close(fig)
    print("Saved roc_curve_test.png")

    # ============================================================
    # 4. PROBE-SCORE (LOGIT) DISTRIBUTION ON TEST
    # ============================================================
    logits_test = X_test_s @ w + b   # (500,)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(logits_test[y_test == 0], bins=40, alpha=0.6,
            label="negative", color="#d62728")
    ax.hist(logits_test[y_test == 1], bins=40, alpha=0.6,
            label="positive", color="#1f77b4")
    ax.axvline(0, color="black", linestyle="--", lw=1, label="decision boundary")
    ax.set_xlabel("probe score  (w · x + b)")
    ax.set_ylabel("count")
    ax.set_title("Test-set probe scores by true label\n(separation = linear sentiment direction works)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "score_distribution_test.png", dpi=150)
    plt.close(fig)
    print("Saved score_distribution_test.png")

    # ============================================================
    # 5. WEIGHT-MAGNITUDE HISTOGRAM + TOP DIMENSIONS
    # ============================================================
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(np.abs(w), bins=60, color="#2ca02c", alpha=0.85)
    ax.set_xlabel("|w_i|  (weight magnitude per dimension)")
    ax.set_ylabel("count")
    ax.set_title(f"Probe weight magnitudes across {len(w)} dimensions  "
                 f"(‖w‖₂ = {np.linalg.norm(w):.3f})")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "weight_magnitudes.png", dpi=150)
    plt.close(fig)
    print("Saved weight_magnitudes.png")

    top_pos = np.argsort(w)[-20:][::-1]
    top_neg = np.argsort(w)[:20]
    top_dims = {
        "top_20_positive_weight_dims": [
            {"dim": int(d), "weight": float(w[d])} for d in top_pos
        ],
        "top_20_negative_weight_dims": [
            {"dim": int(d), "weight": float(w[d])} for d in top_neg
        ],
        "weight_norm_l2": float(np.linalg.norm(w)),
        "weight_mean_abs": float(np.mean(np.abs(w))),
        "weight_max_abs": float(np.max(np.abs(w))),
    }
    (RESULTS_DIR / "top_weight_dimensions.json").write_text(
        json.dumps(top_dims, indent=2)
    )
    print(f"\nTop 5 positive-weight dimensions:")
    for d in top_pos[:5]:
        print(f"  dim {int(d):4d}  w = {w[d]:+.4f}")
    print(f"Top 5 negative-weight dimensions:")
    for d in top_neg[:5]:
        print(f"  dim {int(d):4d}  w = {w[d]:+.4f}")

    # ============================================================
    # 6. QUALITATIVE INSPECTION
    # ============================================================
    pred_test = clf.predict(X_test_s)
    correct = pred_test == y_test

    # Most confident correct positives: pred=1, true=1, highest logit
    pos_correct = np.where(correct & (y_test == 1))[0]
    neg_correct = np.where(correct & (y_test == 0))[0]
    wrong = np.where(~correct)[0]

    def take(idxs, by, k=5):
        order = np.argsort(by[idxs])
        return idxs[order[-k:][::-1]]

    top_pos_correct = take(pos_correct, logits_test, 5)
    top_neg_correct = take(neg_correct, -logits_test, 5)   # most negative logits
    # Worst mistakes: largest |logit| with wrong prediction
    worst_wrong = wrong[np.argsort(-np.abs(logits_test[wrong]))][:5]

    def fmt(idx_list):
        return [
            {
                "idx": int(i),
                "true": int(y_test[i]),
                "pred": int(pred_test[i]),
                "logit": float(logits_test[i]),
                "p_pos": float(proba_test[i]),
                "sentence": test_rows[i]["sentence"],
            }
            for i in idx_list
        ]

    qualitative = {
        "most_confident_correct_positive": fmt(top_pos_correct),
        "most_confident_correct_negative": fmt(top_neg_correct),
        "worst_mistakes": fmt(worst_wrong),
    }
    (RESULTS_DIR / "probe_qualitative.json").write_text(
        json.dumps(qualitative, indent=2)
    )

    print("\n=== QUALITATIVE EXAMPLES (test set) ===")
    print("\nMost-confident correct POSITIVES:")
    for ex in qualitative["most_confident_correct_positive"]:
        print(f"  logit={ex['logit']:+.2f}  p={ex['p_pos']:.3f}  {ex['sentence']!r}")
    print("\nMost-confident correct NEGATIVES:")
    for ex in qualitative["most_confident_correct_negative"]:
        print(f"  logit={ex['logit']:+.2f}  p={ex['p_pos']:.3f}  {ex['sentence']!r}")
    print("\nWorst MISTAKES (high confidence, wrong label):")
    for ex in qualitative["worst_mistakes"]:
        print(f"  logit={ex['logit']:+.2f}  p={ex['p_pos']:.3f}  "
              f"true={ex['true']} pred={ex['pred']}  {ex['sentence']!r}")

    print("\nAll analysis artefacts written to results/ and results/plots/.")


if __name__ == "__main__":
    main()
