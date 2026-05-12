"""
Step 3a (pilot): Run the full activation-extraction logic on just 10 training
sentences. Prints diagnostics. Saves nothing.

The point is to surface every silent bug — hook placement, pooling axis,
tokenization quirks, dtype, label alignment — in 30 seconds, before we
commit to a 10-minute run on all 3000 examples.

Run from project root:
    source .venv/bin/activate
    python scripts/02a_pilot_extraction.py
"""

import json
import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-2-2b"
TARGET_LAYER = 12
N_PILOT = 10

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_JSON = PROJECT_ROOT / "data" / "train_samples.json"


def main() -> None:
    # --- 1. Auth + device ---
    load_dotenv(PROJECT_ROOT / ".env")
    login(token=os.environ["HUGGING_FACE_TOKEN"], add_to_git_credential=False)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"[1/6] Device: {device}")

    # --- 2. Load model + tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16
    ).to(device)
    model.eval()
    hidden_size = model.config.hidden_size
    n_layers = len(model.model.layers)
    print(f"[2/6] Model loaded. layers={n_layers}, hidden={hidden_size}")

    # --- 3. Load 10 pilot sentences ---
    all_rows = json.loads(TRAIN_JSON.read_text())
    rows = all_rows[:N_PILOT]
    print(f"[3/6] Pilot: {len(rows)} sentences")
    for r in rows:
        print(f"      [{r['label']}] {r['sentence']!r}")

    # --- 4. Hook layer 12 residual stream output ---
    captured = {}

    def hook(_m, _i, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["hidden"] = hidden.detach()

    handle = model.model.layers[TARGET_LAYER].register_forward_hook(hook)

    # --- 5. Inspect tokenization for the FIRST sentence (sanity) ---
    first = rows[0]["sentence"]
    enc = tokenizer(first, return_tensors="pt")
    ids = enc["input_ids"][0].tolist()
    decoded = [tokenizer.decode([t]) for t in ids]
    print(f"[5/6] First sentence tokenization:")
    print(f"      sentence: {first!r}")
    print(f"      n_tokens: {len(ids)}")
    print(f"      ids:      {ids}")
    print(f"      decoded:  {decoded}")

    # --- 6. Run all 10 sentences and collect mean-pooled activations ---
    pooled_vectors = []
    raw_shapes = []
    with torch.no_grad():
        for r in rows:
            inputs = tokenizer(r["sentence"], return_tensors="pt").to(device)
            model(**inputs)
            h = captured["hidden"]                  # (1, seq_len, hidden)
            raw_shapes.append(tuple(h.shape))
            pooled = h.mean(dim=1).squeeze(0)       # (hidden,)
            pooled_vectors.append(pooled.float().cpu())
    handle.remove()

    pooled = torch.stack(pooled_vectors)            # (N_PILOT, hidden)
    labels = torch.tensor([r["label"] for r in rows])

    print(f"[6/6] Diagnostics:")
    print(f"      raw shapes (per sentence): {raw_shapes}")
    print(f"      pooled tensor shape:        {tuple(pooled.shape)}  "
          f"(expected ({N_PILOT}, {hidden_size}))")
    print(f"      pooled dtype:               {pooled.dtype}")
    print(f"      labels:                     {labels.tolist()}")
    print(f"      any NaN?                    {torch.isnan(pooled).any().item()}")
    print(f"      any inf?                    {torch.isinf(pooled).any().item()}")
    print(f"      mean / std / min / max:     "
          f"{pooled.mean():.4f} / {pooled.std():.4f} / "
          f"{pooled.min():.4f} / {pooled.max():.4f}")
    print(f"      first vector first 6 dims:  {pooled[0, :6].tolist()}")

    # Per-class mean magnitude — extremely rough sanity that the two classes
    # are not identical at the activation level. With only 10 samples this is
    # noisy; we just want to confirm the values aren't *bit-for-bit* equal.
    pos = pooled[labels == 1]
    neg = pooled[labels == 0]
    if len(pos) > 0 and len(neg) > 0:
        diff = (pos.mean(0) - neg.mean(0)).abs().mean().item()
        print(f"      |mean(pos) - mean(neg)|.mean(): {diff:.4f}  "
              "(should be > 0; meaningful magnitude in step 3c on full data)")

    # Hard assertions
    assert pooled.shape == (N_PILOT, hidden_size)
    assert pooled.dtype == torch.float32
    assert not torch.isnan(pooled).any()
    assert not torch.isinf(pooled).any()
    print("\nAll pilot checks passed.")


if __name__ == "__main__":
    main()
