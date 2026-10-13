"""
Confusion matrices on the test set for the no-BOS linear probe, the top-100 SAE
classifier and the single-feature threshold classifier (layer 12).

Writes results/confusion_matrices_nobos.json and
       results/plots/confusion_matrices_nobos.png   (Appendix C figure)
"""
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

y = np.load("data/SAEs/test_labels.npy")
clf = pickle.load(open("models/linear_probe_nobos.pkl", "rb"))
sc = pickle.load(open("models/scaler_nobos.pkl", "rb"))
X = np.load("data/linear_probes/all_layers_nobos_test.npy", mmap_mode="r")[:, 12]
p_pred = clf.predict(sc.transform(np.asarray(X)))

corrs = np.load("results/train_sae_feature_correlations_max.npy")
idx = np.argsort(-np.abs(corrs))[:100]
gtr = np.load("data/SAEs/train_sae_features_max.npy")
gte = np.load("data/SAEs/test_sae_features_max.npy")
ytr = np.load("data/SAEs/train_labels.npy")
s = StandardScaler()
c = LogisticRegression(C=1.0, max_iter=2000, random_state=42).fit(s.fit_transform(gtr[:, idx]), ytr)
s_pred = c.predict(s.transform(gte[:, idx]))
cut = json.load(open("results/sae_cutoff.json"))["cutoff"]
f_pred = (gte[:, 14733] <= cut).astype(int)

sns.set_theme(style="white", context="talk")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
out = {}
for ax, (name, pred) in zip(axes, [("Linear probe", p_pred), ("SAE top-100 classifier", s_pred),
                                   ("SAE single feature (threshold)", f_pred)]):
    cm = confusion_matrix(y, pred)
    acc = float((pred == y).mean())
    out[name] = {"confusion_matrix": cm.tolist(), "accuracy": acc,
                 "negative_recall": float(cm[0, 0] / cm[0].sum()), "positive_recall": float(cm[1, 1] / cm[1].sum())}
    print(f"{name:32s} acc={acc:.3f} cm={cm.tolist()}")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, annot_kws={"size": 20},
                xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"], vmin=0, vmax=280)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True" if ax is axes[0] else "")
    ax.set_title(f"{name}\n(accuracy {acc*100:.1f}%)", fontsize=15)
plt.tight_layout()
plt.savefig("results/plots/confusion_matrices_nobos.png", dpi=160, bbox_inches="tight")
json.dump(out, open("results/confusion_matrices_nobos.json", "w"), indent=2)
