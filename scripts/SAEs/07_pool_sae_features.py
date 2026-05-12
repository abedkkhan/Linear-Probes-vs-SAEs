"""
Step 5 (Script B2, revised): Pool per-token SAE features across tokens.
BOS was already removed in Script B1, so this is a plain mean/max pool.

Reads:   data/SAEs/{train,val,test}_token_sae_features.npy   object array
Writes:  data/SAEs/{train,val,test}_sae_features_mean.npy    (N, 16384)
         data/SAEs/{train,val,test}_sae_features_max.npy     (N, 16384)

Run from project root:
    source .venv/bin/activate
    python scripts/SAEs/07_pool_sae_features.py
"""

import os
import time

import numpy as np

SPLITS = ["train", "val", "test"]
DATA_DIR = "data/SAEs"
D_SAE = 16384


def main() -> None:
    for split in SPLITS:
        in_path = os.path.join(DATA_DIR, f"{split}_token_sae_features.npy")
        token_feats = np.load(in_path, allow_pickle=True)
        N = len(token_feats)
        print(f"\n[{split}] {N} sentences loaded from {in_path}")

        feats_mean = np.zeros((N, D_SAE), dtype=np.float32)
        feats_max  = np.zeros((N, D_SAE), dtype=np.float32)
        token_counts = []

        t0 = time.time()
        for i in range(N):
            t = token_feats[i].astype(np.float32)   # (T, 16384), BOS-free
            token_counts.append(t.shape[0])
            feats_mean[i] = t.mean(axis=0)
            feats_max[i]  = t.max(axis=0)
        elapsed = time.time() - t0

        tc = np.array(token_counts)
        dead_mean = int(((feats_mean != 0).sum(axis=0) == 0).sum())
        dead_max  = int(((feats_max  != 0).sum(axis=0) == 0).sum())

        out_mean = os.path.join(DATA_DIR, f"{split}_sae_features_mean.npy")
        out_max  = os.path.join(DATA_DIR, f"{split}_sae_features_max.npy")
        np.save(out_mean, feats_mean)
        np.save(out_max,  feats_max)

        print(f"[{split}] Pooled in {elapsed:.1f}s")
        print(f"        tokens per sentence: "
              f"mean={tc.mean():.1f}  min={int(tc.min())}  max={int(tc.max())}")
        print(f"        dead features in mean-pool = {dead_mean} / {D_SAE}")
        print(f"        dead features in max-pool  = {dead_max} / {D_SAE}")
        print(f"        saved -> {out_mean}")
        print(f"        saved -> {out_max}")

    print("\nAll splits pooled.")


if __name__ == "__main__":
    main()