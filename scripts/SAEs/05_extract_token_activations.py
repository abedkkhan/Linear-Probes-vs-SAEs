"""
Step 5 (Script A): Extract per-token layer-12 activations for all 3000
sentences — NO mean pooling. Each sentence's activations are saved as a
(T, 2304) array, and all sentences in a split are stored together as a
NumPy object array.

Reads:   data/{train,val,test}_samples.json
Writes:  data/SAEs/{train,val,test}_token_activations.npy   object array,
                                                            entry i has
                                                            shape (T_i, 2304)
         data/SAEs/{train,val,test}_labels.npy              (N,)

Run from project root:
    source .venv/bin/activate
    python scripts/SAEs/05_extract_token_activations.py
"""

import json
import os
import sys
import time

import numpy as np
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-2-2b"
TARGET_LAYER = 12

SPLITS = ["train", "val", "test"]
IN_DIR = "data"
OUT_DIR = "data/SAEs"


def main() -> None:
    # --- 1. Auth ---
    load_dotenv()
    token = os.getenv("HUGGING_FACE_TOKEN")
    if not token:
        sys.exit("ERROR: HUGGING_FACE_TOKEN not found in .env")
    login(token=token, add_to_git_credential=False)
    print("[1/4] Logged in to Hugging Face.")

    # --- 2. Device ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[2/4] Using device: {device}")

    # --- 3. Load Gemma + hook layer 12 ---
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to(device)
    model.eval()
    print(f"[3/4] Gemma loaded in {time.time() - t0:.1f}s")

    captured = {}

    def hook(_m, _i, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["hidden"] = hidden.detach()

    handle = model.model.layers[TARGET_LAYER].register_forward_hook(hook)
    print(f"      Hook attached to layer {TARGET_LAYER}.")

    # --- 4. Process each split ---
    os.makedirs(OUT_DIR, exist_ok=True)

    for split in SPLITS:
        in_path = os.path.join(IN_DIR, f"{split}_samples.json")
        with open(in_path) as f:
            rows = json.load(f)
        N = len(rows)
        print(f"\n[{split}] {N} sentences")

        # Object array: each entry will hold one sentence's (T_i, 2304) array.
        token_acts = np.empty(N, dtype=object)
        labels = np.zeros(N, dtype=np.int64)
        token_counts = []

        t0 = time.time()
        with torch.no_grad():
            for i, row in enumerate(rows):
                inputs = tokenizer(row["sentence"], return_tensors="pt").to(device)
                model(**inputs)

                # captured["hidden"] is (1, T, 2304) fp16 on MPS
                h = captured["hidden"].squeeze(0)              # (T, 2304)
                h_np = h.to(torch.float32).cpu().numpy()       # save as fp32

                token_acts[i] = h_np
                labels[i] = int(row["label"])
                token_counts.append(h_np.shape[0])

                if (i + 1) % 250 == 0:
                    print(f"        {i+1}/{N}  ({time.time() - t0:.1f}s)")

        elapsed = time.time() - t0

        # Save
        out_acts = os.path.join(OUT_DIR, f"{split}_token_activations.npy")
        out_lbls = os.path.join(OUT_DIR, f"{split}_labels.npy")
        np.save(out_acts, token_acts, allow_pickle=True)
        np.save(out_lbls, labels)

        tc = np.array(token_counts)
        print(f"[{split}] Done in {elapsed:.1f}s")
        print(f"        token counts: min={tc.min()}  max={tc.max()}  "
              f"mean={tc.mean():.1f}  total={tc.sum()}")
        print(f"        first sentence shape = {token_acts[0].shape}, "
              f"dtype = {token_acts[0].dtype}")
        print(f"        saved -> {out_acts}")
        print(f"        saved -> {out_lbls}")

    handle.remove()
    print("\n[4/4] All splits saved.")


if __name__ == "__main__":
    main()