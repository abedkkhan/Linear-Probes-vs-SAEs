"""
Phase 5 + Phase 6 plots: visualise SAE experiment results and compare against
the linear probe baseline. All figures saved to results/plots/.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")

DATA_DIR = "data/SAEs"
RESULTS_DIR = "results"
PLOTS_DIR = "results/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load everything
# ---------------------------------------------------------------------------

with open(os.path.join(RESULTS_DIR, "probe_metrics.json")) as f:
    probe = json.load(f)
with open(os.path.join(RESULTS_DIR, "sae_classifier_metrics.json")) as f:
    sae_single = json.load(f)
with open(os.path.join(RESULTS_DIR, "sae_cutoff.json")) as f:
    sae_cutoff = json.load(f)
with open(os.path.join(RESULTS_DIR, "sae_topk_metrics.json")) as f:
    sae_topk = json.load(f)
with open(os.path.join(RESULTS_DIR, "direction_comparison.json")) as f:
    direction = json.load(f)
with open(os.path.join(RESULTS_DIR, "train_top_sae_features_max.json")) as f:
    top_corr = json.load(f)

corrs = np.load(os.path.join(RESULTS_DIR, "train_sae_feature_correlations_max.npy"))

# ---------------------------------------------------------------------------
# Figure 1 — Headline: SAE top-K vs linear probe (test accuracy)
# ---------------------------------------------------------------------------

Ks = sae_topk["K_values"]
test_acc = [sae_topk["results"][str(k)]["test"]["accuracy"] for k in Ks]
val_acc  = [sae_topk["results"][str(k)]["val"]["accuracy"]  for k in Ks]
probe_test = probe["test"]["accuracy"]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(Ks, [a*100 for a in test_acc], "o-", linewidth=2.5, markersize=8,
        color="#1f77b4", label="SAE top-K probe (test)")
ax.plot(Ks, [a*100 for a in val_acc], "s--", linewidth=1.5, markersize=6,
        color="#1f77b4", alpha=0.45, label="SAE top-K probe (val)")
ax.axhline(probe_test * 100, color="#d62728", linewidth=2.5, linestyle="-",
           label=f"Linear probe (test = {probe_test*100:.1f}%)")
ax.axhline(50, color="grey", linewidth=1, linestyle=":", label="Random (50%)")

ax.set_xscale("log")
ax.set_xlabel("K (number of top SAE features used)")
ax.set_ylabel("Accuracy (%)")
ax.set_title("SAE classifier accuracy vs. linear probe\n(Gemma 2 2B, layer 12, SST-2)")
ax.set_ylim(45, 100)
ax.legend(loc="lower right", fontsize=12)
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "sae_topk_vs_probe.png"), dpi=160, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Figure 2 — Method comparison bar chart
# ---------------------------------------------------------------------------

labels = ["Random", "1 SAE feat", "5 SAE feats", "10 SAE feats",
          "50 SAE feats", "100 SAE feats", "Linear probe (2304-d)"]
values = [50.0,
          sae_topk["results"]["1"]["test"]["accuracy"] * 100,
          sae_topk["results"]["5"]["test"]["accuracy"] * 100,
          sae_topk["results"]["10"]["test"]["accuracy"] * 100,
          sae_topk["results"]["50"]["test"]["accuracy"] * 100,
          sae_topk["results"]["100"]["test"]["accuracy"] * 100,
          probe_test * 100]
colors = ["grey", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#d62728"]

fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.8)
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.5, f"{v:.1f}%",
            ha="center", va="bottom", fontsize=12, fontweight="bold")
ax.set_ylabel("Test accuracy (%)")
ax.set_title("Test-set accuracy: SAE feature classifiers vs. linear probe")
ax.set_ylim(40, 100)
ax.axhline(probe_test * 100, color="#d62728", linewidth=1, linestyle=":", alpha=0.5)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "accuracy_comparison_bar.png"), dpi=160, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Figure 3 — Top 20 SAE features by |training correlation|
# ---------------------------------------------------------------------------

top20 = top_corr[:20]
ids = [str(e["feature_id"]) for e in top20]
cors = [e["correlation"] for e in top20]
clr  = ["#d62728" if e["fires_on"] == "negative" else "#2ca02c" for e in top20]

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(range(len(top20)), cors, color=clr, edgecolor="black", linewidth=0.6)
ax.set_xticks(range(len(top20)))
ax.set_xticklabels(ids, rotation=60, fontsize=10)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Training correlation with label")
ax.set_xlabel("SAE feature ID")
ax.set_title("Top 20 SAE features by |correlation| with sentiment label\n"
             "(red = fires on negative, green = fires on positive)")
ax.grid(True, alpha=0.3, axis="y")

# Annotate the headline feature
headline_idx = ids.index("14733")
ax.annotate("feature 14733\n(headline)",
            xy=(headline_idx, cors[headline_idx]),
            xytext=(headline_idx + 1.5, cors[headline_idx] - 0.08),
            fontsize=11, ha="left",
            arrowprops=dict(arrowstyle="->", color="black"))
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "sae_feature_correlations.png"), dpi=160, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Figure 4 — Cosine alignment of top features with probe direction w
# ---------------------------------------------------------------------------

# Align in order of top_corr's top-20
align_map = {e["feature_id"]: e["cosine_with_probe_w"] for e in direction["top_by_correlation"]}
cosines = [align_map[e["feature_id"]] for e in top20]

rand_abs_mean = direction["random_baseline"]["mean_abs"]
rand_std      = direction["random_baseline"]["std_signed"]

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(range(len(top20)), cosines, color=clr, edgecolor="black", linewidth=0.6)
ax.set_xticks(range(len(top20)))
ax.set_xticklabels(ids, rotation=60, fontsize=10)
ax.axhline(0, color="black", linewidth=0.8)
ax.axhspan(-rand_std*2, rand_std*2, color="grey", alpha=0.15,
           label=f"Random-feature ±2σ band (|cos|≈{rand_abs_mean:.3f})")
ax.set_ylabel("cos(probe w, SAE decoder)")
ax.set_xlabel("SAE feature ID")
ax.set_title("Cosine similarity between probe direction w and SAE decoder vectors\n"
             "(top 20 features by training correlation)")
ax.legend(loc="lower right", fontsize=11)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "sae_cosine_alignment.png"), dpi=160, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Figure 5 — Correlation vs cosine scatter (internal consistency)
# ---------------------------------------------------------------------------

all_corr_abs = np.abs(corrs)
# Compute cosines for all features for the scatter — use the decoder matrix
# saved implicitly via direction comparison. We don't have W_dec here; use the
# overlap of features we DO have cosines for (top 20 + top alignment 20).
known_fids = {e["feature_id"]: e["cosine_with_probe_w"] for e in direction["top_by_correlation"]}
known_fids.update({e["feature_id"]: e["cosine_with_probe_w"] for e in direction["top_by_alignment"]})
xs = [all_corr_abs[fid] for fid in known_fids]
ys = [abs(c) for c in known_fids.values()]

fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(xs, ys, s=70, alpha=0.7, edgecolor="black", linewidth=0.6, color="#1f77b4")
ax.set_xlabel("|training correlation with label|")
ax.set_ylabel("|cosine with probe direction w|")
ax.set_title("Internal consistency: features with high label-correlation\n"
             "also tend to have decoder vectors aligned with probe w")
ax.axhline(rand_abs_mean, color="grey", linestyle=":", linewidth=1, label="random |cos| baseline")
ax.legend(loc="lower right", fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "sae_correlation_vs_alignment.png"), dpi=160, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Figure 6 — Distribution of feature 14733 max-pool activations
# ---------------------------------------------------------------------------

X_test = np.load(os.path.join(DATA_DIR, "test_sae_features_max.npy"))
y_test = np.load(os.path.join(DATA_DIR, "test_labels.npy"))
f_vals = X_test[:, 14733]
cutoff = sae_cutoff["cutoff"]

fig, ax = plt.subplots(figsize=(11, 6))
bins = np.linspace(0, max(f_vals.max(), cutoff*1.5), 50)
ax.hist(f_vals[y_test == 1], bins=bins, alpha=0.65, color="#2ca02c",
        edgecolor="black", linewidth=0.4, label=f"Positive sentences (n={(y_test==1).sum()})")
ax.hist(f_vals[y_test == 0], bins=bins, alpha=0.65, color="#d62728",
        edgecolor="black", linewidth=0.4, label=f"Negative sentences (n={(y_test==0).sum()})")
ax.axvline(cutoff, color="black", linewidth=2.5, linestyle="--",
           label=f"Cutoff = {cutoff:.2f} (from val)")
ax.set_xlabel("Feature 14733 max-pool activation")
ax.set_ylabel("Number of test sentences")
ax.set_title("Distribution of feature 14733's activations on test\n"
             "(feature fires on negative sentiment — predicts negative when above cutoff)")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "feature_14733_distribution.png"), dpi=160, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Figure 7 — Side-by-side confusion matrices
# ---------------------------------------------------------------------------

cm_probe = np.array(probe["test"]["confusion_matrix"])
cm_sae   = np.array(sae_single["test"]["confusion_matrix"])
cm_topk  = np.array(sae_topk["results"]["100"]["test"]["confusion_matrix"])

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
titles = [
    f"Linear probe ({probe['test']['accuracy']*100:.1f}%)",
    f"SAE feature 14733 ({sae_single['test']['accuracy']*100:.1f}%)",
    f"SAE top-100 probe ({sae_topk['results']['100']['test']['accuracy']*100:.1f}%)",
]
for ax, cm, title in zip(axes, [cm_probe, cm_sae, cm_topk], titles):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["pred neg", "pred pos"],
                yticklabels=["true neg", "true pos"],
                ax=ax, annot_kws={"fontsize": 18, "fontweight": "bold"})
    ax.set_title(title, fontsize=14)
fig.suptitle("Test-set confusion matrices", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "confusion_comparison.png"), dpi=160, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Figure 8 — Histogram of all 16,384 feature correlations
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 6))
ax.hist(corrs, bins=80, color="#4292c6", edgecolor="black", linewidth=0.4)
ax.axvline(0, color="black", linewidth=0.8)
top_corr_val = top_corr[0]["correlation"]
ax.axvline(top_corr_val, color="#d62728", linewidth=2, linestyle="--",
           label=f"Top feature (14733): {top_corr_val:.3f}")
ax.set_xlabel("Correlation with sentiment label (training set)")
ax.set_ylabel("Number of SAE features")
ax.set_title("Distribution of training correlations across all 16,384 SAE features\n"
             "Most features carry no sentiment signal; a small tail aligns with sentiment")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "all_feature_correlations_hist.png"), dpi=160, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Figure 9 — F1 and AUC comparison
# ---------------------------------------------------------------------------

methods = ["1 SAE feat", "5 feats", "10 feats", "50 feats", "100 feats", "Linear probe"]
test_f1 = [
    sae_topk["results"]["1"]["test"]["f1"],
    sae_topk["results"]["5"]["test"]["f1"],
    sae_topk["results"]["10"]["test"]["f1"],
    sae_topk["results"]["50"]["test"]["f1"],
    sae_topk["results"]["100"]["test"]["f1"],
    probe["test"]["f1"],
]
test_auc = [
    sae_topk["results"]["1"]["test"]["auc"],
    sae_topk["results"]["5"]["test"]["auc"],
    sae_topk["results"]["10"]["test"]["auc"],
    sae_topk["results"]["50"]["test"]["auc"],
    sae_topk["results"]["100"]["test"]["auc"],
    probe["test"]["auc_roc"],
]
x = np.arange(len(methods))
w_bar = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
b1 = ax.bar(x - w_bar/2, test_f1, w_bar, color="#1f77b4", edgecolor="black",
            linewidth=0.6, label="F1")
b2 = ax.bar(x + w_bar/2, test_auc, w_bar, color="#ff7f0e", edgecolor="black",
            linewidth=0.6, label="AUC")
for bars in (b1, b2):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=15, ha="right")
ax.set_ylim(0.5, 1.02)
ax.set_ylabel("Test-set score")
ax.set_title("F1 and AUC across SAE configurations and linear probe")
ax.legend(loc="lower right", fontsize=12)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "f1_auc_comparison.png"), dpi=160, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
print(f"\nAll plots saved to: {PLOTS_DIR}")
for name in sorted(os.listdir(PLOTS_DIR)):
    if name.endswith(".png"):
        size = os.path.getsize(os.path.join(PLOTS_DIR, name)) / 1024
        print(f"  {name}  ({size:.0f} KB)")
