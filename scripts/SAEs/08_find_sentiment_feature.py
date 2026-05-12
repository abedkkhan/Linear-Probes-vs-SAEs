"""
Step 5d: For every SAE feature, compute its correlation with the sentiment
label. Rank features by |correlation|. Save and print the top candidates.

We run this on both pooling strategies (mean and max) so we can compare.

Reads:   data/SAEs/train_sae_features_{mean,max}.npy   (2000, 16384)
         data/SAEs/train_labels.npy                    (2000,)
Writes:  results/sae_feature_correlations_{mean,max}.npy   (16384,)
         results/top_sae_features_{mean,max}.json          top-50 ranked

Run from project root:
    source .venv/bin/activate
    python scripts/SAEs/08_find_sentiment_feature.py
"""

"""
Step 5d: For every SAE feature, compute its correlation with the sentiment
label. Rank features by |correlation|. Save and print the top candidates.

We run this on both pooling strategies (mean and max) so we can compare.

Reads:   data/SAEs/train_sae_features_{mean,max}.npy   (2000, 16384)
         data/SAEs/train_labels.npy                    (2000,)
Writes:  results/sae_feature_correlations_{mean,max}.npy   (16384,)
         results/top_sae_features_{mean,max}.json          top-50 ranked

Run from project root:
    source .venv/bin/activate
    python scripts/SAEs/08_find_sentiment_feature.py
"""

import json
import os

import numpy as np

DATA_DIR = "data/SAEs"
OUT_DIR = "results"
POOLINGS = ["mean", "max"]
TOP_K = 50

def column_correlations(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson correlation of every column of X with y. Returns (16384,)."""
    y = y.astype(np.float64)
    X = X.astype(np.float64)

    y_centered = y - y.mean()
    X_centered = X - X.mean(axis=0, keepdims=True)

    num = (X_centered * y_centered.reshape(-1, 1)).sum(axis=0)
    denom = np.sqrt((X_centered ** 2).sum(axis=0) * (y_centered ** 2).sum())

    # Dead features have zero variance; safely set their correlation to 0.
    corrs = np.zeros_like(num)
    nonzero = denom > 0
    corrs[nonzero] = num[nonzero] / denom[nonzero]
    return corrs

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    y = np.load(os.path.join(DATA_DIR, "train_labels.npy"))   # (2000,)
    print(f"Labels: {y.shape}  pos={int((y==1).sum())}  neg={int((y==0).sum())}")

    for pool in POOLINGS:
        feats_path = os.path.join(DATA_DIR, f"train_sae_features_{pool}.npy")
        X = np.load(feats_path)                               # (2000, 16384)
        print(f"\n[{pool}-pool] features: {X.shape}")

        corrs = column_correlations(X, y)                     # (16384,)

        # Rank by absolute correlation
        order = np.argsort(-np.abs(corrs))
        top_idx = order[:TOP_K]
        top = [
            {
                "rank": int(r + 1),
                "feature_id": int(idx),
                "correlation": float(corrs[idx]),
                "fires_on": "positive" if corrs[idx] > 0 else "negative",
                "fraction_nonzero": float((X[:, idx] != 0).mean()),
            }
            for r, idx in enumerate(top_idx)
        ]

        np.save(os.path.join(OUT_DIR, f"train_sae_feature_correlations_{pool}.npy"), corrs)
        with open(os.path.join(OUT_DIR, f"train_top_sae_features_{pool}.json"), "w") as f:
              json.dump(top, f, indent=2)
            

        # Print top 20
        print(f"[{pool}-pool] top 20 by |corr|:")
        print(f"{'rank':>4} {'feat':>6} {'corr':>8} {'sign':>10} {'nonzero%':>9}")
        for entry in top[:20]:
            print(f"{entry['rank']:>4} {entry['feature_id']:>6} "
                  f"{entry['correlation']:>+8.4f} {entry['fires_on']:>10} "
                  f"{entry['fraction_nonzero']*100:>8.1f}%")

        # Sanity stats
        n_strong = int((np.abs(corrs) > 0.3).sum())
        print(f"[{pool}-pool] features with |corr| > 0.3: {n_strong}")
        print(f"[{pool}-pool] features with |corr| > 0.5: {int((np.abs(corrs) > 0.5).sum())}")

    print("\nDone.")


if __name__ == "__main__":
    main()