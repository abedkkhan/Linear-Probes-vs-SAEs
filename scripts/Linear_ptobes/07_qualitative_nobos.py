"""
Qualitative comparison of the no-BOS linear probe and the SAE classifiers on
the layer-12 test set. CPU only; uses cached activations and SAE features.

Writes results/qualitative_nobos.json with:
  - probe: most confident errors (false positives / false negatives)
  - agreement table between probe and SAE top-100 classifier
  - example sentences where only the probe / only the SAE is correct
  - sentences both get wrong
"""

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data"
RES = ROOT / "results"

rows = json.loads((DATA / "test_samples.json").read_text())
sent = [r["sentence"] for r in rows]
y = np.load(DATA / "SAEs" / "test_labels.npy")

# probe (no BOS)
clf = pickle.load(open(ROOT / "models" / "linear_probe_nobos.pkl", "rb"))
sc = pickle.load(open(ROOT / "models" / "scaler_nobos.pkl", "rb"))
X = np.load(DATA / "linear_probes" / "all_layers_nobos_test.npy", mmap_mode="r")[:, 12]
p_prob = clf.predict_proba(sc.transform(np.asarray(X)))[:, 1]
p_pred = (p_prob >= 0.5).astype(int)

# SAE top-100 classifier (same recipe as 11_topk_sae_probe.py)
corrs = np.load(RES / "train_sae_feature_correlations_max.npy")
rank = np.argsort(-np.abs(corrs))[:100]
gtr = np.load(DATA / "SAEs" / "train_sae_features_max.npy")[:, rank]
gte = np.load(DATA / "SAEs" / "test_sae_features_max.npy")
ytr = np.load(DATA / "SAEs" / "train_labels.npy")
s2 = StandardScaler()
c2 = LogisticRegression(C=1.0, max_iter=2000, random_state=42).fit(s2.fit_transform(gtr), ytr)
s_prob = c2.predict_proba(s2.transform(gte[:, rank]))[:, 1]
s_pred = (s_prob >= 0.5).astype(int)
assert abs((s_pred == y).mean() - 0.898) < 1e-9, (s_pred == y).mean()

# single feature threshold classifier
cut = json.load(open(RES / "sae_cutoff.json"))["cutoff"]
f1433 = gte[:, 14733]
f_pred = (f1433 <= cut).astype(int)

pc, sc_ = p_pred == y, s_pred == y


def item(i):
    return {"sentence": sent[i], "label": "positive" if y[i] else "negative",
            "probe_p_pos": round(float(p_prob[i]), 3), "sae100_p_pos": round(float(s_prob[i]), 3),
            "feat14733": round(float(f1433[i]), 2)}


fp = [i for i in np.argsort(-p_prob) if y[i] == 0 and p_pred[i] == 1][:8]
fn = [i for i in np.argsort(p_prob) if y[i] == 1 and p_pred[i] == 0][:8]
only_probe = [i for i in np.argsort(-np.abs(s_prob - 0.5)) if pc[i] and not sc_[i]][:8]
only_sae = [i for i in np.argsort(-np.abs(p_prob - 0.5)) if sc_[i] and not pc[i]][:8]
both_wrong = [i for i in range(len(y)) if not pc[i] and not sc_[i]]

out = {
    "probe_test_acc": float(pc.mean()), "sae100_test_acc": float(sc_.mean()),
    "single_feature_test_acc": float((f_pred == y).mean()),
    "agreement": {"both_correct": int((pc & sc_).sum()), "probe_only": int((pc & ~sc_).sum()),
                  "sae_only": int((~pc & sc_).sum()), "both_wrong": int((~pc & ~sc_).sum()),
                  "prediction_agreement": float((p_pred == s_pred).mean())},
    "probe_errors": {"false_positive": int(((p_pred == 1) & (y == 0)).sum()),
                     "false_negative": int(((p_pred == 0) & (y == 1)).sum())},
    "probe_most_confident_false_positives": [item(i) for i in fp],
    "probe_most_confident_false_negatives": [item(i) for i in fn],
    "probe_correct_sae_wrong": [item(i) for i in only_probe],
    "sae_correct_probe_wrong": [item(i) for i in only_sae],
    "both_wrong": [item(i) for i in both_wrong],
}
(RES / "qualitative_nobos.json").write_text(json.dumps(out, indent=2))
print(json.dumps({k: out[k] for k in ["probe_test_acc", "sae100_test_acc", "single_feature_test_acc",
                                      "agreement", "probe_errors"]}, indent=2))
for key in ["probe_most_confident_false_positives", "probe_most_confident_false_negatives",
            "probe_correct_sae_wrong", "sae_correct_probe_wrong", "both_wrong"]:
    print(f"\n== {key} ==")
    for it in out[key]:
        print(f"  [{it['label'][:3]}] probe={it['probe_p_pos']:.2f} sae={it['sae100_p_pos']:.2f} "
              f"f={it['feat14733']:5.2f} | {it['sentence']}")
