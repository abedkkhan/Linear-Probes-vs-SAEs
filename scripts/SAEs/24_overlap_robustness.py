"""
Robustness check for phrase overlap between the training and test sets.

Both sets are sampled from the official SST-2 training split, which contains
phrases extracted from the same reviews. This script counts the test sentences
that share a phrase of at least MIN_WORDS words with a training sentence, and
recomputes test accuracy (layer 12) with those sentences excluded. It also
counts the short fragments among the sentences on which only one method is
correct.

Writes results/overlap_robustness.json.
Run from project root after 06_probe_pipeline_nobos.py and the SAE pipeline (05-11).
"""
import json
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

MIN_WORDS = 5

train = [d["sentence"].strip() for d in json.load(open("data/train_samples.json"))]
test = [d["sentence"].strip() for d in json.load(open("data/test_samples.json"))]
y = np.load("data/SAEs/test_labels.npy")


def shares_phrase(t, r):
    """True if one sentence contains the other and the shared phrase has >= MIN_WORDS words."""
    if t == r:
        return True
    return (t in r and len(t.split()) >= MIN_WORDS) or (r in t and len(r.split()) >= MIN_WORDS)


overlap = np.array([any(shares_phrase(t, r) for r in train) for t in test])

# linear probe (no BOS)
clf = pickle.load(open("models/linear_probe_nobos.pkl", "rb"))
sc = pickle.load(open("models/scaler_nobos.pkl", "rb"))
X = np.load("data/linear_probes/all_layers_nobos_test.npy", mmap_mode="r")[:, 12]
probe_ok = clf.predict(sc.transform(np.asarray(X))) == y

# SAE top-100 classifier (same recipe as 11_topk_sae_probe.py)
corrs = np.load("results/train_sae_feature_correlations_max.npy")
idx = np.argsort(-np.abs(corrs))[:100]
gtr = np.load("data/SAEs/train_sae_features_max.npy")
gte = np.load("data/SAEs/test_sae_features_max.npy")
ytr = np.load("data/SAEs/train_labels.npy")
s = StandardScaler()
c = LogisticRegression(C=1.0, max_iter=2000, random_state=42).fit(s.fit_transform(gtr[:, idx]), ytr)
sae_ok = c.predict(s.transform(gte[:, idx])) == y

# single-feature threshold classifier
cut = json.load(open("results/sae_cutoff.json"))["cutoff"]
single_ok = (gte[:, 14733] <= cut).astype(int) == y


def acc(mask):
    return {"n": int(mask.sum()), "probe": float(probe_ok[mask].mean()),
            "sae_top100": float(sae_ok[mask].mean()), "sae_single": float(single_ok[mask].mean())}


n_words = np.array([len(t.split()) for t in test])
probe_only, sae_only = probe_ok & ~sae_ok, sae_ok & ~probe_ok
res = {
    "min_shared_words": MIN_WORDS,
    "test_sentences_sharing_phrase_with_train": int(overlap.sum()),
    "accuracy_all": acc(np.ones(len(test), bool)),
    "accuracy_without_overlap": acc(~overlap),
    "accuracy_overlap_only": acc(overlap),
    "disagreements": {
        "probe_only_correct": int(probe_only.sum()),
        "probe_only_correct_three_words_or_fewer": int((probe_only & (n_words <= 3)).sum()),
        "sae_only_correct": int(sae_only.sum()),
        "sae_only_correct_three_words_or_fewer": int((sae_only & (n_words <= 3)).sum()),
    },
}
json.dump(res, open("results/overlap_robustness.json", "w"), indent=2)
print(json.dumps(res, indent=2))
