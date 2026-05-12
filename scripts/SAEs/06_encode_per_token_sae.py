"""
Step 5 (Script B1, revised): Drop the BOS token, then encode the remaining
per-token Gemma activations through the SAE. Save per-token SAE features
(content tokens only — BOS already removed).

Reads:   data/SAEs/{train,val,test}_token_activations.npy   object array
                                                            entry i: (T_i, 2304)
Writes:  data/SAEs/{train,val,test}_token_sae_features.npy  object array
                                                            entry i: (T_i - 1, 16384)
                                                            BOS removed.

Run from project root:
    source .venv/bin/activate
    python scripts/SAEs/06_encode_per_token_sae.py
"""

import os
import sys
import time

import numpy as np
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from sae_lens import SAE

SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID = "layer_12/width_16k/canonical"

SPLITS = ["train", "val", "test"]
DATA_DIR = "data/SAEs"


def main() -> None:
    load_dotenv()
    token = os.getenv("HUGGING_FACE_TOKEN")
    if not token:
        sys.exit("ERROR: HUGGING_FACE_TOKEN not found in .env")
    login(token=token, add_to_git_credential=False)
    print("[1/3] Logged in to Hugging Face.")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    t0 = time.time()
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device=str(device))
    sae.eval()
    print(f"[2/3] SAE loaded in {time.time() - t0:.1f}s "
          f"(d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae})")

    for split in SPLITS:
        in_path = os.path.join(DATA_DIR, f"{split}_token_activations.npy")
        acts = np.load(in_path, allow_pickle=True)
        N = len(acts)
        print(f"\n[{split}] {N} sentences loaded from {in_path}")

        token_feats = np.empty(N, dtype=object)
        l0_list = []

        t0 = time.time()
        with torch.no_grad():
            for i in range(N):
                h_np = acts[i]                                # (T, 2304)

                # Drop BOS (index 0). Keep at least one token in fallback.
                if h_np.shape[0] > 1:
                    h_np = h_np[1:]                           # (T-1, 2304)

                h = torch.from_numpy(h_np).to(device=device, dtype=torch.float32)
                feats = sae.encode(h)                         # (T-1, 16384)

                token_feats[i] = feats.to(torch.float16).cpu().numpy()
                l0_list.append((feats != 0).float().sum(dim=1).cpu().numpy())

                if (i + 1) % 500 == 0:
                    print(f"        {i+1}/{N}  ({time.time() - t0:.1f}s)")

        elapsed = time.time() - t0
        all_l0 = np.concatenate(l0_list)

        out_path = os.path.join(DATA_DIR, f"{split}_token_sae_features.npy")
        np.save(out_path, token_feats, allow_pickle=True)
        size_mb = os.path.getsize(out_path) / (1024 * 1024)

        print(f"[{split}] Done in {elapsed:.1f}s")
        print(f"        content-token L0:  mean={all_l0.mean():.1f}  "
              f"min={int(all_l0.min())}  max={int(all_l0.max())}")
        print(f"        first sentence shape = {token_feats[0].shape}")
        print(f"        saved -> {out_path}  ({size_mb:.1f} MB)")

    print("\n[3/3] All splits encoded (BOS-free).")


if __name__ == "__main__":
    main()