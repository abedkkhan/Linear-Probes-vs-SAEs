"""
Step 1: Load SST-2, inspect it, and carve out the deterministic subsets used
throughout the experiment.

Why this is its own script:
- The official SST-2 `test` split on HuggingFace has labels = -1 (GLUE
  leaderboard hides them). We cannot use it for evaluation.
- Standard workaround: carve our train+test from the original `train` split,
  and use the original `validation` split as our `val`.

Outputs (saved to data/):
- train_samples.json  — list of {idx, sentence, label} dicts (2000)
- val_samples.json    — 500
- test_samples.json   — 500
- subset_manifest.json — exact source indices + seed for reproducibility

Run from project root:
    source .venv/bin/activate
    python scripts/01_prepare_dataset.py
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset

DATASET_ID = "stanfordnlp/sst2"
SEED = 42
N_TRAIN = 2000
N_VAL = 500
N_TEST = 500

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Load full dataset ---
    print(f"[1/5] Loading {DATASET_ID} ...")
    ds = load_dataset(DATASET_ID)
    print(f"      splits: { {k: len(v) for k, v in ds.items()} }")
    print(f"      columns: {ds['train'].column_names}")
    print(f"      label feature: {ds['train'].features['label']}")

    # --- 2. Sanity-check splits ---
    train_labels = ds["train"]["label"]
    val_labels = ds["validation"]["label"]
    test_labels = ds["test"]["label"]
    print(f"[2/5] Label distributions:")
    print(f"      train      {Counter(train_labels)}")
    print(f"      validation {Counter(val_labels)}")
    print(f"      test       {Counter(test_labels)}  "
          "<-- -1 means hidden labels, unusable for eval")

    # Show 3 sample rows so we know what we're feeding the model.
    print(f"[3/5] First 3 training examples:")
    for i in range(3):
        ex = ds["train"][i]
        print(f"      [{ex['label']}] {ex['sentence'].strip()!r}")

    # --- 4. Deterministic sampling ---
    rng = np.random.default_rng(SEED)

    # Train + test from the original `train` split (no overlap).
    n_source_train = len(ds["train"])
    perm = rng.permutation(n_source_train)
    train_idx = sorted(perm[:N_TRAIN].tolist())
    test_idx = sorted(perm[N_TRAIN:N_TRAIN + N_TEST].tolist())
    assert len(set(train_idx) & set(test_idx)) == 0, "train/test overlap!"

    # Val from the original `validation` split (already held out from training).
    n_source_val = len(ds["validation"])
    val_idx = sorted(
        rng.permutation(n_source_val)[:N_VAL].tolist()
    )

    # --- 5. Materialise & save ---
    def collect(split_name: str, indices: list[int]) -> list[dict]:
        rows = []
        for i in indices:
            ex = ds[split_name][i]
            rows.append({
                "source_idx": int(i),
                "sentence": ex["sentence"].strip(),
                "label": int(ex["label"]),
            })
        return rows

    train_rows = collect("train", train_idx)
    test_rows = collect("train", test_idx)
    val_rows = collect("validation", val_idx)

    for name, rows in [("train", train_rows),
                       ("val", val_rows),
                       ("test", test_rows)]:
        path = DATA_DIR / f"{name}_samples.json"
        path.write_text(json.dumps(rows, indent=2))
        labels = Counter(r["label"] for r in rows)
        print(f"      saved {path.name:20s}  n={len(rows):4d}  labels={dict(labels)}")

    manifest = {
        "dataset_id": DATASET_ID,
        "seed": SEED,
        "splits": {
            "train": {"source_split": "train",
                      "n": N_TRAIN,
                      "source_indices": train_idx},
            "val":   {"source_split": "validation",
                      "n": N_VAL,
                      "source_indices": val_idx},
            "test":  {"source_split": "train",
                      "n": N_TEST,
                      "source_indices": test_idx},
        },
        "note": ("Train+test carved from the original train split because the "
                 "official test split has hidden labels (-1)."),
    }
    (DATA_DIR / "subset_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"      saved subset_manifest.json")

    print("\nDataset preparation complete.")


if __name__ == "__main__":
    main()
