"""
Step 0: Verify Gemma 2 2B can be authenticated, downloaded, loaded onto MPS,
and produces activations of the expected shape.

Run from project root:
    source .venv/bin/activate
    python scripts/00_verify_model_load.py
"""

import os
import sys
import time

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-2-2b"
TARGET_LAYER = 12  # zero-indexed residual stream output


def main() -> None:
    # --- 1. Auth ---
    load_dotenv()
    token = os.getenv("HUGGING_FACE_TOKEN")
    if not token:
        sys.exit("ERROR: HUGGING_FACE_TOKEN not found in .env")
    login(token=token, add_to_git_credential=False)
    print("[1/5] Logged in to Hugging Face.")

    # --- 2. Device selection ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[2/5] Using device: {device}")

    # --- 3. Tokenizer ---
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"[3/5] Tokenizer loaded in {time.time() - t0:.1f}s "
          f"(vocab={tokenizer.vocab_size})")

    # --- 4. Model (fp16, weights download on first run, ~5 GB) ---
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
    ).to(device)
    model.eval()
    print(f"[4/5] Model loaded in {time.time() - t0:.1f}s")

    n_layers = len(model.model.layers)
    hidden_size = model.config.hidden_size
    print(f"      num_hidden_layers = {n_layers}")
    print(f"      hidden_size       = {hidden_size}")
    print(f"      target layer {TARGET_LAYER} module: "
          f"{type(model.model.layers[TARGET_LAYER]).__name__}")

    # --- 5. One forward pass with a hook on layer-12 residual stream output ---
    captured = {}

    def hook(_module, _inputs, output):
        # Decoder layers return a tuple; first element is the residual stream.
        hidden = output[0] if isinstance(output, tuple) else output
        captured["hidden"] = hidden.detach()

    handle = model.model.layers[TARGET_LAYER].register_forward_hook(hook)

    sentence = "A wonderful little production."
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        model(**inputs)
    handle.remove()

    h = captured["hidden"]
    print(f"[5/5] Forward pass OK. Layer {TARGET_LAYER} activation shape: "
          f"{tuple(h.shape)}  (expected: (1, seq_len, {hidden_size}))")
    assert h.shape[-1] == hidden_size, "hidden dim mismatch"
    assert h.shape[0] == 1, "batch dim mismatch"

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
