"""
Layer sweep (probe-only): addresses the assessor's IMPORTANT comment on why
Layer 12 was chosen and whether results are sensitive to layer choice.

We hook ALL 26 decoder layers in a single forward pass per sentence, mean-pool
each layer's residual stream over the sequence (matching the original probe
methodology exactly — all tokens included), then train an independent linear
probe at every layer and record train/val/test accuracy, F1, and AUC.

Methodology is identical to scripts/Linear_ptobes/03_train_linear_probe.py:
  - mean pool over all tokens
  - StandardScaler fit on train, applied to val/test
  - LogisticRegression(C=1.0, max_iter=1000, random_state=42, penalty="l2")

Outputs:
    data/linear_probes/all_layers_{split}.npy   (N, 26, 2304) fp32
    results/layer_sweep_metrics.json
    results/plots/layer_sweep_accuracy.png

Run from project root:
    source .venv/bin/activate
    python scripts/Linear_ptobes/05_layer_sweep.py
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-2-2b"
N_LAYERS = 26
SPLITS = ("train", "val", "test")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LP_DATA_DIR = DATA_DIR / "linear_probes"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"


def extract_all_layers(model, tokenizer, device, rows, hidden_size):
    """Return (N, 26, hidden) mean-pooled activations + (N,) labels."""
    captured = {}

    def make_hook(idx):
        def hook(_m, _i, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[idx] = hidden.detach()
        return hook

    handles = [model.model.layers[i].register_forward_hook(make_hook(i))
               for i in range(N_LAYERS)]

    N = len(rows)
    acts = np.empty((N, N_LAYERS, hidden_size), dtype=np.float32)
    labels = np.empty(N, dtype=np.int64)

    try:
        with torch.no_grad():
            for i, row in enumerate(tqdm(rows, desc="extract")):
                inputs = tokenizer(row["sentence"], return_tensors="pt").to(device)
                model(**inputs)
                for L in range(N_LAYERS):
                    h = captured[L]                       # (1, T, hidden)
                    acts[i, L] = h.mean(dim=1).squeeze(0).float().cpu().numpy()
                labels[i] = row["label"]
    finally:
        for h in handles:
            h.remove()
    return acts, labels


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    login(token=os.environ["HUGGING_FACE_TOKEN"], add_to_git_credential=False)
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to(device)
    model.eval()
    hidden = model.config.hidden_size
    print(f"Model ready in {time.time()-t0:.1f}s (layers={len(model.model.layers)}, hidden={hidden})")

    LP_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # --- Extract (reuse cache if present) ---
    data = {}
    for split in SPLITS:
        cache = LP_DATA_DIR / f"all_layers_{split}.npy"
        lbl_cache = LP_DATA_DIR / f"all_layers_{split}_labels.npy"
        if cache.exists() and lbl_cache.exists():
            print(f"[{split}] loading cached {cache.name}")
            data[split] = (np.load(cache), np.load(lbl_cache))
            continue
        rows = json.loads((DATA_DIR / f"{split}_samples.json").read_text())
        print(f"\n[{split}] {len(rows)} sentences — extracting all {N_LAYERS} layers")
        t0 = time.time()
        acts, labels = extract_all_layers(model, tokenizer, device, rows, hidden)
        print(f"  done in {time.time()-t0:.1f}s  shape={acts.shape}")
        np.save(cache, acts)
        np.save(lbl_cache, labels)
        data[split] = (acts, labels)

    Xtr_all, ytr = data["train"]
    Xva_all, yva = data["val"]
    Xte_all, yte = data["test"]

    # --- Train a probe at every layer ---
    print("\nTraining a probe at each layer...")
    results = {}
    for L in range(N_LAYERS):
        Xtr, Xva, Xte = Xtr_all[:, L], Xva_all[:, L], Xte_all[:, L]
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42, penalty="l2", solver="lbfgs")
        clf.fit(Xtr_s, ytr)

        def ev(X, y):
            Xs = scaler.transform(X)
            pred = clf.predict(Xs)
            proba = clf.predict_proba(Xs)[:, 1]
            return (float(accuracy_score(y, pred)),
                    float(f1_score(y, pred)),
                    float(roc_auc_score(y, proba)))

        tr_acc = float(accuracy_score(ytr, clf.predict(Xtr_s)))
        va = ev(Xva, yva)
        te = ev(Xte, yte)
        results[str(L)] = {
            "train_acc": tr_acc,
            "val_acc": va[0], "val_f1": va[1], "val_auc": va[2],
            "test_acc": te[0], "test_f1": te[1], "test_auc": te[2],
        }
        print(f"  layer {L:2d}  train={tr_acc:.3f}  val={va[0]:.3f}  test={te[0]:.3f}  test_auc={te[2]:.3f}")

    # Best layer by validation accuracy (honest model-selection)
    best_layer = max(range(N_LAYERS), key=lambda L: results[str(L)]["val_acc"])
    print(f"\nBest layer by validation accuracy: {best_layer} "
          f"(val={results[str(best_layer)]['val_acc']:.3f}, "
          f"test={results[str(best_layer)]['test_acc']:.3f})")
    print(f"Layer 12 (our choice): "
          f"val={results['12']['val_acc']:.3f}, test={results['12']['test_acc']:.3f}")

    out = {
        "n_layers": N_LAYERS,
        "best_layer_by_val": best_layer,
        "chosen_layer": 12,
        "results": results,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "layer_sweep_metrics.json").write_text(json.dumps(out, indent=2))
    print(f"Saved {RESULTS_DIR/'layer_sweep_metrics.json'}")

    # --- Plot accuracy vs layer ---
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="talk")

    layers = list(range(N_LAYERS))
    tr = [results[str(L)]["train_acc"] * 100 for L in layers]
    va = [results[str(L)]["val_acc"] * 100 for L in layers]
    te = [results[str(L)]["test_acc"] * 100 for L in layers]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(layers, tr, "o-", color="#9ecae1", linewidth=1.5, markersize=5, label="train")
    ax.plot(layers, va, "s--", color="#6baed6", linewidth=1.5, markersize=5, label="val")
    ax.plot(layers, te, "o-", color="#1f77b4", linewidth=2.5, markersize=7, label="test")
    ax.axvline(12, color="#d62728", linewidth=2, linestyle=":", label="layer 12 (chosen)")
    ax.axvline(best_layer, color="#2ca02c", linewidth=1.5, linestyle="--",
               label=f"best layer ({best_layer})")
    ax.set_xlabel("Layer (residual stream output)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Linear probe accuracy across all 26 layers (SST-2, Gemma 2 2B)")
    ax.set_xticks(range(0, N_LAYERS, 2))
    ax.legend(loc="lower center", ncol=2, fontsize=11)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_png = PLOTS_DIR / "layer_sweep_accuracy.png"
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
