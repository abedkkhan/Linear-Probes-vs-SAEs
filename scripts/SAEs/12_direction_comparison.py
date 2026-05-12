"""
Phase 6: Compare the linear probe's sentiment direction w with the SAE's
decoder vectors. The probe direction was found via supervised training; the
SAE decoder vectors come from unsupervised dictionary learning. If both
methods agree on what direction sentiment lives in, their cosine similarity
should be high (in absolute value).

Reads:   results/probe_direction.npy                            (2304,)
         results/train_top_sae_features_max.json                top-50 list
Writes:  results/direction_comparison.json
"""

import json
import os
import sys

import numpy as np
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from sae_lens import SAE

SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID = "layer_12/width_16k/canonical"
RESULTS_DIR = "results"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    # --- 1. Auth + device ---
    load_dotenv()
    token = os.getenv("HUGGING_FACE_TOKEN")
    if not token:
        sys.exit("ERROR: HUGGING_FACE_TOKEN not found in .env")
    login(token=token, add_to_git_credential=False)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # --- 2. Load probe direction ---
    w = np.load(os.path.join(RESULTS_DIR, "probe_direction.npy")).astype(np.float64)
    print(f"Probe w: shape={w.shape}  L2 norm={np.linalg.norm(w):.4f}")
    # sklearn convention: coef_[0] is the weight for the POSITIVE class (label 1).
    # So w POINTS TOWARDS positive sentiment.

    # --- 3. Load SAE and grab the decoder matrix ---
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device=str(device))
    sae.eval()
    W_dec = sae.W_dec.detach().cpu().to(torch.float32).numpy()   # (d_sae, d_in) = (16384, 2304)
    print(f"SAE W_dec: shape={W_dec.shape}  dtype={W_dec.dtype}")

    assert W_dec.shape[1] == w.shape[0], "decoder width must match probe direction"

    # --- 4. Cosines with all SAE decoder rows ---
    # Each row of W_dec is one feature's decoder vector in residual-stream space.
    norms = np.linalg.norm(W_dec, axis=1)               # (16384,)
    w_norm = np.linalg.norm(w)
    cos_all = (W_dec @ w) / (norms * w_norm)            # (16384,)

    # --- 5. Top features by correlation (from step 5d) ---
    with open(os.path.join(RESULTS_DIR, "train_top_sae_features_max.json")) as fp:
        top_by_corr = json.load(fp)

    print("\nTop features by training |correlation| — and their decoder cosines with w:")
    print(f"{'rank':>4} {'feat':>6} {'corr':>8} {'fires_on':>9} {'cos(w, W_dec)':>15}")
    detailed = []
    for entry in top_by_corr[:20]:
        fid = entry["feature_id"]
        c = float(cos_all[fid])
        print(f"{entry['rank']:>4} {fid:>6} "
              f"{entry['correlation']:>+8.4f} {entry['fires_on']:>9} {c:>+15.4f}")
        detailed.append({
            "rank_by_corr": entry["rank"],
            "feature_id": fid,
            "correlation": entry["correlation"],
            "fires_on": entry["fires_on"],
            "cosine_with_probe_w": c,
        })

    # --- 6. Top features by absolute cosine alignment with w (regardless of correlation) ---
    top_align_idx = np.argsort(-np.abs(cos_all))[:20]
    print("\nTop features by |cos(W_dec, w)| — most direction-aligned regardless of correlation:")
    print(f"\n{'rank':>4} {'feat':>6} {'cos(w, W_dec)':>15}")
    top_align = []
    for r, fid in enumerate(top_align_idx, start=1):
        print(f"{r:>4} {int(fid):>6} {float(cos_all[fid]):>+15.4f}")
        top_align.append({
            "rank_by_alignment": r,
            "feature_id": int(fid),
            "cosine_with_probe_w": float(cos_all[fid]),
        })

    # --- 7. Headline number: feature 14733 ---
    headline = {
        "feature_id": 14733,
        "training_correlation": float(
            next(e["correlation"] for e in top_by_corr if e["feature_id"] == 14733)
        ),
        "cosine_with_probe_w": float(cos_all[14733]),
    }
    print(f"\nHeadline: feature 14733  "
          f"corr={headline['training_correlation']:+.4f}  "
          f"cos(w, W_dec[14733])={headline['cosine_with_probe_w']:+.4f}")

    # Sanity: random-feature cosine distribution
    rng = np.random.default_rng(42)
    random_idx = rng.choice(W_dec.shape[0], size=1000, replace=False)
    rand_cosines = cos_all[random_idx]
    print(f"\nRandom-feature cosine baseline (1000 features):")
    print(f"  mean={rand_cosines.mean():+.4f}  "
          f"std={rand_cosines.std():.4f}  "
          f"|cos| mean={np.abs(rand_cosines).mean():.4f}")

    out = {
        "headline_feature_14733": headline,
        "top_by_correlation": detailed,
        "top_by_alignment": top_align,
        "random_baseline": {
            "mean_signed": float(rand_cosines.mean()),
            "std_signed": float(rand_cosines.std()),
            "mean_abs": float(np.abs(rand_cosines).mean()),
        },
        "probe_norm": float(w_norm),
        "decoder_norms_summary": {
            "min": float(norms.min()),
            "mean": float(norms.mean()),
            "max": float(norms.max()),
        },
    }
    out_path = os.path.join(RESULTS_DIR, "direction_comparison.json")
    with open(out_path, "w") as fp:
        json.dump(out, fp, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()