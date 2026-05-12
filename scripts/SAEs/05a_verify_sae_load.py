"""
Step 5a: Verify the Gemma Scope SAE for layer 12 (residual stream, width 16K,
canonical sparsity) can be downloaded, loaded onto MPS, and produces sparse
features of the expected shape.

Run from project root:
    source .venv/bin/activate
    python scripts/05a_verify_sae_load.py
"""

import os
import sys
import time

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from sae_lens import SAE

# Gemma Scope SAE coordinates for our experiment.
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID = "layer_12/width_16k/canonical"
EXPECTED_INPUT_DIM = 2304   # Gemma 2 2B residual stream width
EXPECTED_DICT_SIZE = 16384  # 16K features


def main() -> None:
    # --- 1. Auth ---
    load_dotenv()
    token = os.getenv("HUGGING_FACE_TOKEN")
    if not token:
        sys.exit("ERROR: HUGGING_FACE_TOKEN not found in .env")
    login(token=token, add_to_git_credential=False)
    print("[1/4] Logged in to Hugging Face.")

    # --- 2. Device selection ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[2/4] Using device: {device}")

    # --- 3. Load the SAE ---
    # sae-lens looks SAE_RELEASE + SAE_ID up in its catalogue, finds the
    # matching Hugging Face path, downloads weights (~300 MB), caches them,
    # and returns an SAE module ready to run.
    t0 = time.time()
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=SAE_RELEASE,
        sae_id=SAE_ID,
        device=str(device),
    )
    sae.eval()
    print(f"[3/4] SAE loaded in {time.time() - t0:.1f}s")
    print(f"      release  = {SAE_RELEASE}")
    print(f"      sae_id   = {SAE_ID}")
    print(f"      d_in     = {sae.cfg.d_in}        (expected {EXPECTED_INPUT_DIM})")
    print(f"      d_sae    = {sae.cfg.d_sae}      (expected {EXPECTED_DICT_SIZE})")
    print(f"      dtype    = {sae.dtype}")
    print(f"      device   = {sae.device}")

    assert sae.cfg.d_in == EXPECTED_INPUT_DIM, "input dim mismatch"
    assert sae.cfg.d_sae == EXPECTED_DICT_SIZE, "dictionary size mismatch"

    # --- 4. Smoke test: encode one fake activation and check sparsity ---
    # We make a (1, 2304) tensor of random values and pass it through the
    # encoder. The output should be (1, 16384) with most entries == 0.
    fake_activation = torch.randn(1, EXPECTED_INPUT_DIM, device=device, dtype=sae.dtype)
    with torch.no_grad():
        features = sae.encode(fake_activation)

    n_total = features.numel()
    n_nonzero = (features != 0).sum().item()
    pct_nonzero = 100.0 * n_nonzero / n_total

    print(f"[4/4] Encode smoke test:")
    print(f"      output shape    = {tuple(features.shape)}  (expected (1, {EXPECTED_DICT_SIZE}))")
    print(f"      non-zero feats  = {n_nonzero} / {n_total}  ({pct_nonzero:.2f}%)")
    assert features.shape == (1, EXPECTED_DICT_SIZE), "feature shape mismatch"

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
