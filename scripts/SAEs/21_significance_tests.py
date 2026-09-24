"""
Exact McNemar tests comparing the no-BOS linear probe with the SAE classifiers
on paired test-set predictions (layer 12), plus a 95% CI on probe accuracy.

Writes results/significance_tests_nobos.json.
Run from project root after 06_probe_pipeline_nobos.py and the SAE pipeline (05-11).
"""
import json
import pickle

import numpy as np
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

y = np.load("data/SAEs/test_labels.npy")
clf = pickle.load(open("models/linear_probe_nobos.pkl", "rb"))
sc = pickle.load(open("models/scaler_nobos.pkl", "rb"))
X = np.load("data/linear_probes/all_layers_nobos_test.npy", mmap_mode="r")[:, 12]
probe_ok = clf.predict(sc.transform(np.asarray(X))) == y

corrs = np.load("results/train_sae_feature_correlations_max.npy")
gtr = np.load("data/SAEs/train_sae_features_max.npy")
gte = np.load("data/SAEs/test_sae_features_max.npy")
ytr = np.load("data/SAEs/train_labels.npy")


def topk_correct(K):
    idx = np.argsort(-np.abs(corrs))[:K]
    s = StandardScaler()
    c = LogisticRegression(C=1.0, max_iter=2000, random_state=42).fit(s.fit_transform(gtr[:, idx]), ytr)
    return c.predict(s.transform(gte[:, idx])) == y


cut = json.load(open("results/sae_cutoff.json"))["cutoff"]
single_ok = (gte[:, 14733] <= cut).astype(int) == y

res = {}
for name, other in [("top100", topk_correct(100)), ("top5", topk_correct(5)),
                    ("top50", topk_correct(50)), ("single_thr", single_ok)]:
    b, c = int((probe_ok & ~other).sum()), int((~probe_ok & other).sum())
    p = binomtest(b, b + c, 0.5).pvalue
    res[name] = {"acc": float(other.mean()), "probe_only": b, "other_only": c, "mcnemar_exact_p": p}
    print(f"probe ({probe_ok.mean():.3f}) vs {name} ({other.mean():.3f}): "
          f"probe-only={b} other-only={c} exact McNemar p={p:.4g}")
acc = probe_ok.mean()
res["probe_ci95_halfwidth"] = float(1.96 * np.sqrt(acc * (1 - acc) / len(y)))
print(f"probe accuracy {acc:.3f} +/- {res['probe_ci95_halfwidth']:.3f} (95% CI)")
json.dump(res, open("results/significance_tests_nobos.json", "w"), indent=2)
