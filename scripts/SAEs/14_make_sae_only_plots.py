"""
Generate SAE-only standalone plots (no linear-probe baseline) for the
presentation slide showing SAE classifier performance.

Outputs in results/plots/:
    sae_topk_only.png            - top-K accuracy curve (SAE only)
    sae_single_confusion.png     - confusion matrix for feature 14733 alone
    sae_f1_auc_only.png          - F1 / AUC across K (SAE only)
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")

RESULTS = "results"
PLOTS = "results/plots"
os.makedirs(PLOTS, exist_ok=True)

with open(os.path.join(RESULTS, "sae_topk_metrics.json")) as f:
    topk = json.load(f)
with open(os.path.join(RESULTS, "sae_classifier_metrics.json")) as f:
    single = json.load(f)

Ks = topk["K_values"]
test_acc = [topk["results"][str(k)]["test"]["accuracy"] for k in Ks]
test_f1  = [topk["results"][str(k)]["test"]["f1"]       for k in Ks]
test_auc = [topk["results"][str(k)]["test"]["auc"]      for k in Ks]

# -----------------------------------------------------------------
# 1) Top-K test accuracy (SAE only)
# -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(Ks, [a * 100 for a in test_acc], "o-",
        linewidth=2.5, markersize=10, color="#1f77b4")
for k, a in zip(Ks, test_acc):
    ax.annotate(f"{a*100:.1f}%", xy=(k, a*100),
                xytext=(0, 12), textcoords="offset points",
                ha="center", fontsize=11, color="#1f3a5f")
ax.axhline(50, color="grey", linewidth=1, linestyle=":", label="Random (50%)")
ax.set_xscale("log")
ax.set_xlabel("K (number of top SAE features used)")
ax.set_ylabel("Test accuracy (%)")
ax.set_title("SAE classifier — test accuracy grows with K")
ax.set_ylim(45, 100)
ax.legend(loc="lower right", fontsize=12)
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "sae_topk_only.png"), dpi=160, bbox_inches="tight")
plt.close()

# -----------------------------------------------------------------
# 2) Single-feature confusion matrix (SAE only)
# -----------------------------------------------------------------
cm = np.array(single["test"]["confusion_matrix"])
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["pred neg", "pred pos"],
            yticklabels=["true neg", "true pos"],
            annot_kws={"fontsize": 22, "fontweight": "bold"},
            ax=ax)
acc = single["test"]["accuracy"]
ax.set_title(f"SAE single feature #14733 — test (acc = {acc:.3f})", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "sae_single_confusion.png"), dpi=160, bbox_inches="tight")
plt.close()

# -----------------------------------------------------------------
# 3) F1 and AUC across K (SAE only) — grouped bars
# -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(Ks))
w = 0.38
b1 = ax.bar(x - w/2, test_f1,  w, color="#1f77b4", edgecolor="black", linewidth=0.6, label="F1")
b2 = ax.bar(x + w/2, test_auc, w, color="#ff7f0e", edgecolor="black", linewidth=0.6, label="AUC")
for bars in (b1, b2):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels([f"K={k}" for k in Ks])
ax.set_ylim(0.5, 1.02)
ax.set_ylabel("Test-set score")
ax.set_title("SAE classifier — F1 and AUC by number of features used")
ax.legend(loc="lower right", fontsize=12)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "sae_f1_auc_only.png"), dpi=160, bbox_inches="tight")
plt.close()

print("Saved 3 SAE-only plots:")
for name in ["sae_topk_only.png", "sae_single_confusion.png", "sae_f1_auc_only.png"]:
    p = os.path.join(PLOTS, name)
    print(f"  {p}  ({os.path.getsize(p)//1024} KB)")
