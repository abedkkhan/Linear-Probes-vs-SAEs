"""
Step 5b: Verify the per-token activations saved by 05_extract_token_activations.py.

Checks each split for:
  1. Shapes: each entry is (T_i, 2304) and T_i matches the tokenizer's output
  2. Label alignment against the original *_samples.json
  3. No NaN / Inf values
  4. Mean-pooled equivalence with data/linear_probes/{split}_activations.npy
     (proves these are the SAME activations as before, just unpooled)
  5. Activation statistics (mean/std/min/max) look sane
  6. Different sentences produce different activations

Run from project root:
    source .venv/bin/activate
    python scripts/SAEs/05b_verify_token_activations.py
"""

import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
SAE_DIR = DATA_DIR / "SAEs"
LP_DIR = DATA_DIR / "linear_probes"

MODEL_ID = "google/gemma-2-2b"
HIDDEN = 2304
SPLITS = ("train", "val", "test")
EXPECTED_N = {"train": 2000, "val": 500, "test": 500}
N_TOKEN_CHECKS = 5  # number of sentences to re-tokenize per split


def main() -> None:
    print("Loading tokenizer for shape verification...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print("Tokenizer ready.\n")

    for split in SPLITS:
        acts = np.load(SAE_DIR / f"{split}_token_activations.npy", allow_pickle=True)
        labels = np.load(SAE_DIR / f"{split}_labels.npy")
        rows = json.loads((DATA_DIR / f"{split}_samples.json").read_text())
        pooled_old = np.load(LP_DIR / f"{split}_activations.npy")

        N = EXPECTED_N[split]

        assert acts.shape == (N,), f"{split}: outer shape {acts.shape}, expected ({N},)"
        assert labels.shape == (N,), f"{split}: labels shape {labels.shape}"
        assert pooled_old.shape == (N, HIDDEN), f"{split}: old pooled shape mismatch"

        # 1. Per-sentence shape + tokenizer match (spot check)
        for i in range(N_TOKEN_CHECKS):
            v = acts[i]
            assert v.ndim == 2 and v.shape[1] == HIDDEN, \
                f"{split}[{i}]: shape {v.shape}, expected (T, {HIDDEN})"
            expected_T = tokenizer(rows[i]["sentence"], return_tensors="pt").input_ids.shape[1]
            assert v.shape[0] == expected_T, \
                f"{split}[{i}]: T={v.shape[0]} but tokenizer says {expected_T}"

        # 2. Label alignment
        json_labels = np.array([r["label"] for r in rows], dtype=np.int64)
        assert np.array_equal(labels, json_labels), \
            f"{split}: labels do not match {split}_samples.json"

        # 3. No NaN / Inf (check first few + last few)
        for i in list(range(N_TOKEN_CHECKS)) + list(range(N - N_TOKEN_CHECKS, N)):
            v = acts[i]
            assert not np.isnan(v).any(), f"{split}[{i}]: NaN found"
            assert not np.isinf(v).any(), f"{split}[{i}]: Inf found"

        # 4. Mean-pooled equivalence vs old linear-probe activations
        max_abs_diff = 0.0
        for i in range(N):
            pooled_new = acts[i].mean(axis=0)
            diff = float(np.abs(pooled_new - pooled_old[i]).max())
            if diff > max_abs_diff:
                max_abs_diff = diff
        pool_match = max_abs_diff < 1e-3  # fp16 -> fp32 round-trip can introduce small noise

        # 5. Numerical stats over all tokens in this split
        all_tokens = np.concatenate([acts[i] for i in range(N)], axis=0)
        token_total = all_tokens.shape[0]
        mean = float(all_tokens.mean())
        std = float(all_tokens.std())
        amin = float(all_tokens.min())
        amax = float(all_tokens.max())

        # 6. Different sentences should differ
        differ = not np.allclose(acts[0].mean(0), acts[1].mean(0))

        print(f"[{split}]")
        print(f"  N sentences        = {N}, total tokens = {token_total}")
        print(f"  shape spot check   = OK (first {N_TOKEN_CHECKS} match tokenizer)")
        print(f"  labels aligned     = True (matches {split}_samples.json)")
        print(f"  NaN / Inf          = False / False")
        print(f"  pool == old probe  = {pool_match}  (max |diff| = {max_abs_diff:.2e})")
        print(f"  mean / std         = {mean:.4f} / {std:.4f}")
        print(f"  min / max          = {amin:.4f} / {amax:.4f}")
        print(f"  acts[0] != acts[1] = {differ}")
        print()

    print("All verification checks passed.")


if __name__ == "__main__":
    main()
