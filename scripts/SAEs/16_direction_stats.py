"""
Direction statistics (addresses assessor comment 2): give the cosine-similarity
comparison proper statistical grounding instead of a bare number.

Computes:
  1. cos(w, decoder_14733) and its z-score / percentile against a null
     distribution of random unit vectors in R^2304.
  2. Same against an empirical null of all 16,384 SAE decoder directions.
  3. Subspace projection: how much of the probe direction ||w|| is captured by
     the span of the top-K SAE decoder vectors (vs a random-K baseline).

Outputs:
    results/direction_stats.json
    results/plots/direction_subspace_projection.png

Run from project root:
    source .venv/bin/activate
    python scripts/SAEs/16_direction_stats.py
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from sae_lens import SAE

sns.set_theme(style="whitegrid", context="talk")

RESULTS = "results"
PLOTS = "results/plots"
FEATURE_ID = 14733
TOP_K_LIST = [1, 5, 10, 20, 50, 100, 200, 500]


def main():
    os.makedirs(PLOTS, exist_ok=True)
    load_dotenv()
    token = os.getenv("HUGGING_FACE_TOKEN")
    if not token:
        sys.exit("HUGGING_FACE_TOKEN missing")
    login(token=token, add_to_git_credential=False)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # --- Load probe direction + SAE decoder ---
    w = np.load(os.path.join(RESULTS, "probe_direction.npy")).astype(np.float64)
    w_unit = w / np.linalg.norm(w)
    d = w.shape[0]

    sae = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res-canonical",
        sae_id="layer_12/width_16k/canonical",
        device=device,
    )
    sae.eval()
    W_dec = sae.W_dec.detach().cpu().to(torch.float64).numpy()        # (16384, 2304)
    W_unit = W_dec / np.linalg.norm(W_dec, axis=1, keepdims=True)

    # Correlations (to pick top-K)
    corrs = np.load(os.path.join(RESULTS, "train_sae_feature_correlations_max.npy"))
    ranking = np.argsort(-np.abs(corrs))

    # --- 1. Headline cosine + null of random unit vectors ---
    cos_feat = float(W_unit[FEATURE_ID] @ w_unit)

    rng = np.random.default_rng(42)
    M = 100_000
    randv = rng.standard_normal((M, d))
    randv /= np.linalg.norm(randv, axis=1, keepdims=True)
    null_cos = randv @ w_unit                       # cosines of random unit vecs with w
    null_mean = float(null_cos.mean())
    null_std = float(null_cos.std())
    # theoretical std of cos in d dims ~ 1/sqrt(d)
    theo_std = 1.0 / np.sqrt(d)
    z_random = (abs(cos_feat) - abs(null_mean)) / null_std
    pct_random = float((np.abs(null_cos) < abs(cos_feat)).mean() * 100)

    # --- 2. Empirical null: all SAE decoder directions ---
    emp_cos = W_unit @ w_unit                       # (16384,)
    emp_abs_mean = float(np.abs(emp_cos).mean())
    emp_abs_std = float(np.abs(emp_cos).std())
    z_emp = (abs(cos_feat) - emp_abs_mean) / emp_abs_std
    pct_emp = float((np.abs(emp_cos) < abs(cos_feat)).mean() * 100)

    # --- 3. Subspace projection (how much of w lives in top-K SAE span) ---
    def captured_fraction(idx):
        D = W_dec[idx]                              # (K, 2304)
        # least-squares projection of w onto row space of D
        coef, *_ = np.linalg.lstsq(D.T, w, rcond=None)
        w_proj = D.T @ coef
        return float(np.linalg.norm(w_proj) / np.linalg.norm(w))

    proj_top = {}
    proj_rand = {}
    for K in TOP_K_LIST:
        proj_top[K] = captured_fraction(ranking[:K])
        # random-K baseline (averaged over 5 draws)
        vals = []
        for _ in range(5):
            ridx = rng.choice(W_dec.shape[0], size=K, replace=False)
            vals.append(captured_fraction(ridx))
        proj_rand[K] = float(np.mean(vals))

    # --- Report ---
    print(f"cos(w, decoder #{FEATURE_ID})           = {cos_feat:+.4f}")
    print(f"random-unit-vector null:  mean|cos|={abs(null_mean):.4f}  "
          f"std={null_std:.4f}  (theory 1/sqrt(d)={theo_std:.4f})")
    print(f"  -> z = {z_random:.1f},  percentile = {pct_random:.2f}%")
    print(f"empirical SAE-decoder null: mean|cos|={emp_abs_mean:.4f}  std={emp_abs_std:.4f}")
    print(f"  -> z = {z_emp:.1f},  percentile = {pct_emp:.2f}%")
    print("\nSubspace projection (fraction of ||w|| captured):")
    for K in TOP_K_LIST:
        print(f"  K={K:4d}   top-K={proj_top[K]:.3f}   random-K={proj_rand[K]:.3f}")

    out = {
        "feature_id": FEATURE_ID,
        "cosine": cos_feat,
        "random_unit_null": {
            "mean": null_mean, "std": null_std,
            "theoretical_std": float(theo_std),
            "z_score": float(z_random), "percentile": pct_random,
        },
        "empirical_sae_null": {
            "abs_mean": emp_abs_mean, "abs_std": emp_abs_std,
            "z_score": float(z_emp), "percentile": pct_emp,
        },
        "subspace_projection": {
            "K": TOP_K_LIST,
            "top_k_fraction": [proj_top[K] for K in TOP_K_LIST],
            "random_k_fraction": [proj_rand[K] for K in TOP_K_LIST],
        },
    }
    with open(os.path.join(RESULTS, "direction_stats.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {RESULTS}/direction_stats.json")

    # --- Plot subspace projection ---
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(TOP_K_LIST, [proj_top[K]*100 for K in TOP_K_LIST], "o-",
            color="#1f77b4", linewidth=2.5, markersize=8, label="top-K SAE features")
    ax.plot(TOP_K_LIST, [proj_rand[K]*100 for K in TOP_K_LIST], "s--",
            color="#9c9489", linewidth=1.5, markersize=6, label="random-K features")
    ax.set_xscale("log")
    ax.set_xlabel("K (SAE decoder vectors in the subspace)")
    ax.set_ylabel("% of probe direction ‖w‖ captured")
    ax.set_title("How much of the probe direction lives in the SAE feature subspace")
    ax.legend(loc="upper left", fontsize=12)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    out_png = os.path.join(PLOTS, "direction_subspace_projection.png")
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
