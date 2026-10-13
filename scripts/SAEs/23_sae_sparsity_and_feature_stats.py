"""
Sparsity of the layer-12 SAE (mean number of active features per content token)
and test-set activation statistics of feature 14733 quoted in Sections 3.4 and 4.1.4.

Writes results/sae_sparsity_feature_stats.json.
"""
import json

import numpy as np

out = {"l0_per_split": {}}
tot, n = 0, 0
for split in ["train", "val", "test"]:
    arr = np.load(f"data/SAEs/{split}_token_sae_features.npy", allow_pickle=True)
    l0 = np.concatenate([(np.asarray(t) != 0).sum(axis=1) for t in arr])
    out["l0_per_split"][split] = {"mean": float(l0.mean()), "tokens": int(len(l0))}
    tot += l0.sum()
    n += len(l0)
out["mean_l0"] = float(tot / n)
out["fraction_active"] = float(tot / n / 16384)

f = np.load("data/SAEs/test_sae_features_max.npy")[:, 14733]
y = np.load("data/SAEs/test_labels.npy")
cut = json.load(open("results/sae_cutoff.json"))["cutoff"]
out["feature_14733_test"] = {
    "silent_negative": int((f[y == 0] == 0).sum()), "n_negative": int((y == 0).sum()),
    "silent_positive": int((f[y == 1] == 0).sum()), "n_positive": int((y == 1).sum()),
    "min_nonzero_activation": float(f[f > 0].min()), "max_activation": float(f.max()),
    "above_threshold_negative": int((f[y == 0] > cut).sum()),
    "above_threshold_positive": int((f[y == 1] > cut).sum()),
}
print(json.dumps(out, indent=2))
json.dump(out, open("results/sae_sparsity_feature_stats.json", "w"), indent=2)
