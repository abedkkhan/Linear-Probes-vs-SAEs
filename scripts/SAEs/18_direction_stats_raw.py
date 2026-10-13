"""
Direction statistics with the probe direction mapped back to raw activation space.

The probe was trained on standardised activations, so clf.coef_ lives in
standardised coordinates. Since w . ((h - mu) / sigma) = (w / sigma) . h + const,
the equivalent direction in the raw residual stream is w_raw = coef_ / scaler.scale_,
which is the space the SAE decoder vectors live in.

Writes (originals are left untouched):
    results/direction_stats_raw.json
    results/plots/direction_subspace_projection_raw.png

Run from project root:
    python scripts/SAEs/18_direction_stats_raw.py
"""

import json
import os
import pickle
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
N_NULL = 100_000


def captured_fraction(W_dec, idx, w):
    D = W_dec[idx]
    coef, *_ = np.linalg.lstsq(D.T, w, rcond=None)
    return float(np.linalg.norm(D.T @ coef) / np.linalg.norm(w))


def main():
    os.makedirs(PLOTS, exist_ok=True)
    load_dotenv()
    token = os.getenv("HUGGING_FACE_TOKEN")
    if not token:
        sys.exit("HUGGING_FACE_TOKEN missing")
    login(token=token, add_to_git_credential=False)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    with open("models/linear_probe.pkl", "rb") as f:
        clf = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    w_std = clf.coef_[0].astype(np.float64)
    saved = np.load(os.path.join(RESULTS, "probe_direction.npy")).astype(np.float64)
    assert np.allclose(w_std, saved, atol=1e-5), "pickled probe does not match probe_direction.npy"

    sigma = scaler.scale_.astype(np.float64)
    w_raw = w_std / sigma
    d = w_raw.shape[0]

    sae = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res-canonical",
        sae_id="layer_12/width_16k/canonical",
        device=device,
    )
    sae.eval()
    W_dec = sae.W_dec.detach().cpu().to(torch.float64).numpy()
    W_unit = W_dec / np.linalg.norm(W_dec, axis=1, keepdims=True)

    corrs = np.load(os.path.join(RESULTS, "train_sae_feature_correlations_max.npy"))
    ranking = np.argsort(-np.abs(corrs))

    rng = np.random.default_rng(42)
    randv = rng.standard_normal((N_NULL, d))
    randv /= np.linalg.norm(randv, axis=1, keepdims=True)

    out = {"feature_id": FEATURE_ID, "n_random_null_vectors": N_NULL,
           "scaler_sigma": {"min": float(sigma.min()), "median": float(np.median(sigma)),
                            "max": float(sigma.max())},
           "cos_w_std_vs_w_raw": float(w_std @ w_raw / (np.linalg.norm(w_std) * np.linalg.norm(w_raw)))}

    for name, w in [("standardised_w", w_std), ("raw_w", w_raw)]:
        w_unit = w / np.linalg.norm(w)
        cos_feat = float(W_unit[FEATURE_ID] @ w_unit)
        null_cos = randv @ w_unit
        z_random = (abs(cos_feat) - abs(null_cos.mean())) / null_cos.std()
        emp = W_unit @ w_unit
        z_emp = (abs(cos_feat) - np.abs(emp).mean()) / np.abs(emp).std()
        top20 = [{"rank": r + 1, "feature_id": int(f), "correlation": float(corrs[f]),
                  "cosine": float(emp[f])} for r, f in enumerate(ranking[:20])]
        align_rank = int((np.abs(emp) > abs(cos_feat)).sum()) + 1
        proj_top, proj_rand = [], []
        for K in TOP_K_LIST:
            proj_top.append(captured_fraction(W_dec, ranking[:K], w))
            proj_rand.append(float(np.mean([
                captured_fraction(W_dec, rng.choice(W_dec.shape[0], size=K, replace=False), w)
                for _ in range(5)])))
        out[name] = {
            "cosine_feature": cos_feat,
            "random_unit_null": {"mean": float(null_cos.mean()), "std": float(null_cos.std()),
                                 "z_score": float(z_random),
                                 "percentile": float((np.abs(null_cos) < abs(cos_feat)).mean() * 100)},
            "empirical_sae_null": {"abs_mean": float(np.abs(emp).mean()),
                                   "abs_std": float(np.abs(emp).std()),
                                   "z_score": float(z_emp),
                                   "percentile": float((np.abs(emp) < abs(cos_feat)).mean() * 100)},
            "feature_rank_by_abs_alignment": align_rank,
            "top20_by_correlation": top20,
            "subspace_projection": {"K": TOP_K_LIST, "top_k_fraction": proj_top,
                                    "random_k_fraction": proj_rand},
        }
        print(f"\n=== {name} ===")
        print(f"cos(w, decoder #{FEATURE_ID}) = {cos_feat:+.4f}   "
              f"z(random)={z_random:.1f}   z(SAE null)={z_emp:.1f}   "
              f"alignment rank among 16384 = {align_rank}")
        for K, a, b in zip(TOP_K_LIST, proj_top, proj_rand):
            print(f"  K={K:4d}  top-K={a:.3f}  random-K={b:.3f}")

    print(f"\ncos(w_std, w_raw) = {out['cos_w_std_vs_w_raw']:.4f}")
    print(f"scaler sigma: min={sigma.min():.4f} median={np.median(sigma):.4f} max={sigma.max():.4f}")

    with open(os.path.join(RESULTS, "direction_stats_raw.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {RESULTS}/direction_stats_raw.json")

    raw = out["raw_w"]["subspace_projection"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(TOP_K_LIST, [v * 100 for v in raw["top_k_fraction"]], "o-",
            color="#1f77b4", linewidth=2.5, markersize=8, label="top-K SAE features")
    ax.plot(TOP_K_LIST, [v * 100 for v in raw["random_k_fraction"]], "s--",
            color="#9c9489", linewidth=1.5, markersize=6, label="random-K features")
    ax.set_xscale("log")
    ax.set_xlabel("K (SAE decoder vectors in the subspace)")
    ax.set_ylabel("% of probe direction ‖w‖ captured")
    ax.set_title("How much of the probe direction lives in the SAE feature subspace")
    ax.legend(loc="upper left", fontsize=12)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    out_png = os.path.join(PLOTS, "direction_subspace_projection_raw.png")
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
