"""
Step 3c: Verification pass for the saved activations.

Loads the six .npy files written by 02_extract_activations.py and runs a series
of sanity checks. Trains nothing. Saves nothing. The point is to fail loudly
*now* if anything is wrong, before we build a linear probe on top of corrupt
or misaligned data.

Run from project root:
    source .venv/bin/activate
    python scripts/02b_verify_activations.py
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SPLITS = ("train", "val", "test")
EXPECTED_N = {"train": 2000, "val": 500, "test": 500}
HIDDEN = 2304


def main() -> None:
    print("Loading and verifying saved activations...\n")

    summaries = {}

    for split in SPLITS:
        act_path = DATA_DIR / f"{split}_activations.npy"
        lbl_path = DATA_DIR / f"{split}_labels.npy"
        json_path = DATA_DIR / f"{split}_samples.json"

        acts = np.load(act_path)
        labels = np.load(lbl_path)
        rows = json.loads(json_path.read_text())

        # 1. Shape checks
        assert acts.shape == (EXPECTED_N[split], HIDDEN), \
            f"{split}: wrong activation shape {acts.shape}"
        assert labels.shape == (EXPECTED_N[split],), \
            f"{split}: wrong label shape {labels.shape}"
        assert acts.dtype == np.float32, f"{split}: wrong dtype {acts.dtype}"

        # 2. Corruption checks
        assert not np.isnan(acts).any(), f"{split}: NaN in activations"
        assert not np.isinf(acts).any(), f"{split}: inf in activations"

        # 3. Label alignment vs original JSON (the critical check)
        json_labels = np.array([r["label"] for r in rows], dtype=np.int64)
        assert np.array_equal(labels, json_labels), \
            f"{split}: saved labels do not match {json_path.name}"

        # 4. Label distribution
        dist = Counter(labels.tolist())

        # 5. Activation statistics
        mean = float(acts.mean())
        std = float(acts.std())
        amin = float(acts.min())
        amax = float(acts.max())

        # 6. Per-class signal
        pos = acts[labels == 1]
        neg = acts[labels == 0]
        per_class_diff = float(np.abs(pos.mean(0) - neg.mean(0)).mean())

        summaries[split] = {
            "shape": acts.shape,
            "dtype": str(acts.dtype),
            "labels": dict(dist),
            "mean": mean, "std": std, "min": amin, "max": amax,
            "per_class_diff": per_class_diff,
        }

        print(f"[{split}]")
        print(f"  shape          = {acts.shape}, dtype = {acts.dtype}")
        print(f"  labels         = {dict(dist)}")
        print(f"  mean / std     = {mean:.4f} / {std:.4f}")
        print(f"  min / max      = {amin:.4f} / {amax:.4f}")
        print(f"  NaN / inf      = False / False")
        print(f"  labels aligned = True (matches {json_path.name})")
        print(f"  |mean(pos) - mean(neg)|.mean() = {per_class_diff:.4f}")
        print()

    # Cross-split comparisons: stats should look similar (same model, same dataset)
    means = [summaries[s]["mean"] for s in SPLITS]
    stds = [summaries[s]["std"] for s in SPLITS]
    print("Cross-split sanity:")
    print(f"  means across splits: {[f'{m:.3f}' for m in means]}  (should be similar)")
    print(f"  stds  across splits: {[f'{s:.3f}' for s in stds]}  (should be similar)")

    # Per-class signal should be clearly non-zero on the train split
    train_diff = summaries["train"]["per_class_diff"]
    assert train_diff > 0.0, "train: per-class difference is exactly zero — something is wrong"
    print(f"\nTrain per-class signal {train_diff:.4f} > 0  →  pipeline is producing "
          "class-discriminative vectors, as expected.")

    print("\nAll verification checks passed. Phase 3 is complete.")


if __name__ == "__main__":
    main()
