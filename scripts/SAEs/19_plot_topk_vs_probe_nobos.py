"""
Top-K SAE classifier accuracy against the no-BOS linear probe baseline.
Same figure as 13_make_plots.py (Figure 1), with the probe line taken from
results/probe_metrics_nobos.json. Writes results/plots/sae_topk_vs_probe_nobos.png.
"""
import json
import os

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")

probe = json.load(open("results/probe_metrics_nobos.json"))
topk = json.load(open("results/sae_topk_metrics.json"))
single = json.load(open("results/sae_classifier_metrics.json"))

Ks = topk["K_values"]
test_acc = [topk["results"][str(k)]["test"]["accuracy"] * 100 for k in Ks]
val_acc = [topk["results"][str(k)]["val"]["accuracy"] * 100 for k in Ks]
probe_test = probe["test"]["accuracy"] * 100

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(Ks, test_acc, "o-", linewidth=2.5, markersize=8, color="#1f77b4", label="SAE top-K classifier (test)")
ax.plot(Ks, val_acc, "s--", linewidth=1.5, markersize=6, color="#1f77b4", alpha=0.45,
        label="SAE top-K classifier (val)")
ax.axhline(probe_test, color="#d62728", linewidth=2.5, label=f"Linear probe (test = {probe_test:.1f}%)")
ax.axhline(single["test"]["accuracy"] * 100, color="#ff7f0e", linewidth=1.5, linestyle="--",
           label=f"Single feature, threshold (test = {single['test']['accuracy']*100:.1f}%)")
ax.axhline(50, color="grey", linewidth=1, linestyle=":", label="Random (50%)")
ax.set_xscale("log")
ax.set_xlabel("K (number of top SAE features used)")
ax.set_ylabel("Accuracy (%)")
ax.set_title("SAE classifier accuracy vs. linear probe\n(Gemma 2 2B, layer 12, SST-2)")
ax.set_ylim(45, 100)
ax.legend(loc="lower right", fontsize=11)
ax.grid(True, alpha=0.4)
plt.tight_layout()
out = os.path.join("results/plots", "sae_topk_vs_probe_nobos.png")
plt.savefig(out, dpi=160, bbox_inches="tight")
print("saved", out)
