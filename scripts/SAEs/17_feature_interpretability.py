"""
Quantitative SAE-feature interpretability (addresses assessor comment 4):
turn "feature 14733 is interpretable" into measurable evidence.

Computes for feature 14733 (and reports top-activating sentences):
  1. Top-activating sentences (train) — what does the feature actually fire on?
  2. Selectivity: mean activation on negative vs positive sentences + a
     selectivity index.
  3. Single-feature precision / recall / F1 for the negative class, at the
     val-chosen cutoff.
  4. Activation density (fraction of sentences where it fires at all).

Outputs:
    results/feature_14733_interpretability.json
    results/plots/feature_14733_top_sentences.png  (text table figure)

Run from project root:
    source .venv/bin/activate
    python scripts/SAEs/17_feature_interpretability.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

DATA = "data"
SAE_DATA = "data/SAEs"
RESULTS = "results"
PLOTS = "results/plots"
FEATURE_ID = 14733


def main():
    os.makedirs(PLOTS, exist_ok=True)

    # --- Load train sentences + max-pooled features + labels ---
    rows = json.loads(open(os.path.join(DATA, "train_samples.json")).read())
    sentences = [r["sentence"] for r in rows]
    X = np.load(os.path.join(SAE_DATA, "train_sae_features_max.npy"))  # (2000, 16384)
    y = np.load(os.path.join(SAE_DATA, "train_labels.npy"))
    f = X[:, FEATURE_ID]

    # --- 1. Top-activating sentences ---
    order = np.argsort(-f)
    top = []
    for idx in order[:12]:
        top.append({
            "activation": float(f[idx]),
            "label": "negative" if y[idx] == 0 else "positive",
            "sentence": sentences[idx][:90],
        })

    print(f"Top-activating sentences for feature {FEATURE_ID}:")
    for t in top[:12]:
        print(f"  {t['activation']:6.2f}  [{t['label'][:3]}]  {t['sentence']}")

    # --- 2. Selectivity ---
    mean_neg = float(f[y == 0].mean())
    mean_pos = float(f[y == 1].mean())
    # selectivity index in [-1, 1]: (neg - pos) / (neg + pos)
    sel_index = (mean_neg - mean_pos) / (mean_neg + mean_pos + 1e-9)

    # --- 3. Precision / recall / F1 for the negative class ---
    cutoff = json.load(open(os.path.join(RESULTS, "sae_cutoff.json")))["cutoff"]
    fires = f > cutoff                       # predicted negative
    is_neg = y == 0
    tp = int(np.sum(fires & is_neg))
    fp = int(np.sum(fires & ~is_neg))
    fn = int(np.sum(~fires & is_neg))
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    # --- 4. Density ---
    density = float((f > 0).mean())
    density_neg = float((f[y == 0] > 0).mean())
    density_pos = float((f[y == 1] > 0).mean())

    print(f"\nSelectivity:  mean(neg)={mean_neg:.3f}  mean(pos)={mean_pos:.3f}  "
          f"index={sel_index:+.3f}")
    print(f"Negative-class @cutoff={cutoff:.2f}:  "
          f"precision={precision:.3f}  recall={recall:.3f}  f1={f1:.3f}")
    print(f"Activation density:  all={density:.3f}  neg={density_neg:.3f}  pos={density_pos:.3f}")

    out = {
        "feature_id": FEATURE_ID,
        "top_activating_sentences": top,
        "selectivity": {
            "mean_activation_negative": mean_neg,
            "mean_activation_positive": mean_pos,
            "selectivity_index": float(sel_index),
        },
        "negative_class_at_cutoff": {
            "cutoff": cutoff,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": tp, "fp": fp, "fn": fn,
        },
        "activation_density": {
            "all": density, "negative": density_neg, "positive": density_pos,
        },
    }
    with open(os.path.join(RESULTS, "feature_14733_interpretability.json"), "w") as fp:
        json.dump(out, fp, indent=2)
    print(f"\nSaved {RESULTS}/feature_14733_interpretability.json")

    # --- Plot: top-activating sentences as a clean table figure ---
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axis("off")
    ax.set_title(f"Feature {FEATURE_ID}: top-activating training sentences",
                 fontsize=15, loc="left", pad=14)

    show = top[:10]
    y0 = 0.93
    dy = 0.092
    ax.text(0.005, y0 + 0.06, "act", fontsize=11, fontweight="bold", color="#5b5b5b")
    ax.text(0.075, y0 + 0.06, "label", fontsize=11, fontweight="bold", color="#5b5b5b")
    ax.text(0.20, y0 + 0.06, "sentence", fontsize=11, fontweight="bold", color="#5b5b5b")
    for i, t in enumerate(show):
        yy = y0 - i * dy
        col = "#d62728" if t["label"] == "negative" else "#2ca02c"
        ax.text(0.005, yy, f"{t['activation']:.1f}", fontsize=11, color="#1a1a1a")
        ax.text(0.075, yy, t["label"], fontsize=11, color=col, fontweight="bold")
        ax.text(0.20, yy, t["sentence"], fontsize=11, color="#1a1a1a")
    plt.tight_layout()
    out_png = os.path.join(PLOTS, "feature_14733_top_sentences.png")
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
