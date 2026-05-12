"""
Step 3b: Full activation extraction.

For each of the three splits (train/val/test):
  1. Load the prepared sentence list from data/{split}_samples.json
  2. For each sentence: tokenize, forward pass through Gemma 2 2B,
     capture layer-12 residual stream, mean-pool across sequence dim
  3. Stack into a (N, 2304) float32 array and save as data/{split}_activations.npy
  4. Save the aligned labels as data/{split}_labels.npy

Reproducibility note: extraction is deterministic given the fixed sentence
list from step 1 — no seeds needed here.

Run from project root:
    source .venv/bin/activate
    python scripts/02_extract_activations.py
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-2-2b"
TARGET_LAYER = 12
SPLITS = ("train", "val", "test")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def main() -> None:
    # --- Auth + device ---
    load_dotenv(PROJECT_ROOT / ".env")
    login(token=os.environ["HUGGING_FACE_TOKEN"], add_to_git_credential=False)
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"Device: {device}")

    # --- Load model + tokenizer once ---
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print("Loading model (fp16)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16
    ).to(device)
    model.eval()
    hidden_size = model.config.hidden_size
    print(f"Model ready in {time.time() - t0:.1f}s "
          f"(layers={len(model.model.layers)}, hidden={hidden_size})")

    # --- Hook on layer-12 residual stream output ---
    captured: dict = {}

    def hook(_m, _i, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["hidden"] = hidden.detach()

    handle = model.model.layers[TARGET_LAYER].register_forward_hook(hook)

    # --- Process each split ---
    try:
        for split in SPLITS:
            sentence_path = DATA_DIR / f"{split}_samples.json"
            rows = json.loads(sentence_path.read_text())
            print(f"\n[{split}] {len(rows)} sentences")

            pooled = np.empty((len(rows), hidden_size), dtype=np.float32)
            labels = np.empty(len(rows), dtype=np.int64)

            t0 = time.time()
            with torch.no_grad():
                for i, row in enumerate(tqdm(rows, desc=f"extract {split}")):
                    inputs = tokenizer(row["sentence"],
                                       return_tensors="pt").to(device)
                    model(**inputs)
                    h = captured["hidden"]                  # (1, T, hidden)
                    vec = h.mean(dim=1).squeeze(0).float().cpu().numpy()
                    pooled[i] = vec
                    labels[i] = row["label"]
            elapsed = time.time() - t0
            print(f"  done in {elapsed:.1f}s "
                  f"({elapsed / len(rows) * 1000:.1f} ms/sentence)")

            # Final integrity check before saving
            assert pooled.shape == (len(rows), hidden_size)
            assert not np.isnan(pooled).any(), f"NaNs in {split}"
            assert not np.isinf(pooled).any(), f"Infs in {split}"

            act_path = DATA_DIR / f"{split}_activations.npy"
            lbl_path = DATA_DIR / f"{split}_labels.npy"
            np.save(act_path, pooled)
            np.save(lbl_path, labels)
            print(f"  saved {act_path.name}  shape={pooled.shape}  "
                  f"dtype={pooled.dtype}  size={act_path.stat().st_size / 1e6:.1f} MB")
            print(f"  saved {lbl_path.name}  shape={labels.shape}")
    finally:
        handle.remove()

    print("\nExtraction complete.")


if __name__ == "__main__":
    main()
