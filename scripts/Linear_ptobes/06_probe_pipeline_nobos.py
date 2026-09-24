"""
Linear-probe pipeline with the BOS token excluded from mean pooling, so the
probe sees exactly the same tokens as the SAE (whose BOS is dropped in
scripts/SAEs/06_encode_per_token_sae.py).

One forward pass per sentence captures all 26 layers. Then:
  1. layer-12 probe (same recipe as 03_train_linear_probe.py)
  2. layer sweep over all 26 layers (same recipe as 05_layer_sweep.py)
  3. direction statistics against the SAE decoder, using the raw-space
     direction w = coef_ / scaler.scale_ (same analysis as SAEs/18)

All outputs carry a _nobos suffix; original results are not touched.

Run from project root:
    python scripts/Linear_ptobes/06_probe_pipeline_nobos.py
"""

import json
import os
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from sae_lens import SAE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-2-2b"
N_LAYERS = 26
TARGET_LAYER = 12
SPLITS = ("train", "val", "test")
FEATURE_ID = 14733
TOP_K_LIST = [1, 5, 10, 20, 50, 100, 200, 500]
N_NULL = 100_000

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data"
LP_DATA = DATA / "linear_probes"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"

sns.set_theme(style="whitegrid", context="talk")


def extract(model, tokenizer, device, rows, hidden):
    captured = {}

    def make_hook(idx):
        def hook(_m, _i, output):
            captured[idx] = (output[0] if isinstance(output, tuple) else output).detach()
        return hook

    handles = [model.model.layers[i].register_forward_hook(make_hook(i)) for i in range(N_LAYERS)]
    acts = np.empty((len(rows), N_LAYERS, hidden), dtype=np.float32)
    labels = np.empty(len(rows), dtype=np.int64)
    try:
        with torch.no_grad():
            for i, row in enumerate(tqdm(rows, desc="extract")):
                inputs = tokenizer(row["sentence"], return_tensors="pt").to(device)
                assert inputs["input_ids"][0, 0].item() == tokenizer.bos_token_id, "first token is not BOS"
                model(**inputs)
                for L in range(N_LAYERS):
                    h = captured[L][0]                      # (T, hidden)
                    h = h[1:] if h.shape[0] > 1 else h      # drop BOS
                    acts[i, L] = h.mean(dim=0).float().cpu().numpy()
                labels[i] = row["label"]
    finally:
        for h in handles:
            h.remove()
    assert np.isfinite(acts).all()
    return acts, labels


def fit_probe(Xtr, ytr):
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42, penalty="l2", solver="lbfgs")
    clf.fit(Xtr_s, ytr)
    return scaler, clf


def metrics(clf, scaler, X, y):
    Xs = scaler.transform(X)
    pred = clf.predict(Xs)
    proba = clf.predict_proba(Xs)[:, 1]
    return {"n": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
            "f1": float(f1_score(y, pred)), "auc_roc": float(roc_auc_score(y, proba)),
            "confusion_matrix": confusion_matrix(y, pred).tolist()}


def captured_fraction(W_dec, idx, w):
    D = W_dec[idx]
    coef, *_ = np.linalg.lstsq(D.T, w, rcond=None)
    return float(np.linalg.norm(D.T @ coef) / np.linalg.norm(w))


def main():
    load_dotenv(ROOT / ".env")
    login(token=os.environ["HUGGING_FACE_TOKEN"], add_to_git_credential=False)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # ---------- 1. extraction (cached) ----------
    LP_DATA.mkdir(parents=True, exist_ok=True)
    data = {}
    model = tokenizer = None
    for split in SPLITS:
        cache = LP_DATA / f"all_layers_nobos_{split}.npy"
        lcache = LP_DATA / f"all_layers_nobos_{split}_labels.npy"
        if cache.exists() and lcache.exists():
            data[split] = (np.load(cache), np.load(lcache))
            print(f"[{split}] cached")
            continue
        if model is None:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to(device)
            model.eval()
        rows = json.loads((DATA / f"{split}_samples.json").read_text())
        t0 = time.time()
        acts, labels = extract(model, tokenizer, device, rows, model.config.hidden_size)
        print(f"[{split}] extracted {acts.shape} in {time.time() - t0:.0f}s")
        np.save(cache, acts)
        np.save(lcache, labels)
        data[split] = (acts, labels)
    if model is not None:
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    Xtr_all, ytr = data["train"]
    Xva_all, yva = data["val"]
    Xte_all, yte = data["test"]

    # sanity: labels must match the original split files
    for split, (_, y) in data.items():
        assert np.array_equal(y, np.load(LP_DATA / f"{split}_labels.npy")) and np.array_equal(y, np.load(DATA / "SAEs" / f"{split}_labels.npy")), f"label mismatch in {split}"

    # ---------- 2. layer-12 probe ----------
    L = TARGET_LAYER
    scaler, clf = fit_probe(Xtr_all[:, L], ytr)
    probe = {"config": {"C": 1.0, "max_iter": 1000, "seed": 42, "penalty": "l2",
                        "solver": "lbfgs", "bos": "excluded"},
             "train": metrics(clf, scaler, Xtr_all[:, L], ytr),
             "val": metrics(clf, scaler, Xva_all[:, L], yva),
             "test": metrics(clf, scaler, Xte_all[:, L], yte)}
    MODELS.mkdir(exist_ok=True)
    with open(MODELS / "linear_probe_nobos.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open(MODELS / "scaler_nobos.pkl", "wb") as f:
        pickle.dump(scaler, f)
    np.save(RESULTS / "probe_direction_nobos.npy", clf.coef_[0].astype(np.float32))
    (RESULTS / "probe_metrics_nobos.json").write_text(json.dumps(probe, indent=2))
    print(f"\nLayer-12 probe (no BOS): test acc={probe['test']['accuracy']:.4f} "
          f"f1={probe['test']['f1']:.4f} auc={probe['test']['auc_roc']:.4f}")

    # ---------- 3. layer sweep ----------
    sweep = {}
    for layer in range(N_LAYERS):
        sc, cl = fit_probe(Xtr_all[:, layer], ytr)
        va = metrics(cl, sc, Xva_all[:, layer], yva)
        te = metrics(cl, sc, Xte_all[:, layer], yte)
        tr_acc = float(accuracy_score(ytr, cl.predict(sc.transform(Xtr_all[:, layer]))))
        sweep[str(layer)] = {"train_acc": tr_acc, "val_acc": va["accuracy"], "val_f1": va["f1"],
                             "val_auc": va["auc_roc"], "test_acc": te["accuracy"],
                             "test_f1": te["f1"], "test_auc": te["auc_roc"]}
        print(f"  layer {layer:2d}  train={tr_acc:.3f}  val={va['accuracy']:.3f}  test={te['accuracy']:.3f}")
    best = max(range(N_LAYERS), key=lambda i: sweep[str(i)]["val_acc"])
    best_test = max(range(N_LAYERS), key=lambda i: sweep[str(i)]["test_acc"])
    (RESULTS / "layer_sweep_metrics_nobos.json").write_text(json.dumps(
        {"n_layers": N_LAYERS, "best_layer_by_val": best, "best_layer_by_test": best_test,
         "chosen_layer": TARGET_LAYER, "bos": "excluded", "results": sweep}, indent=2))
    print(f"Best layer by val: {best}   best by test: {best_test}")

    layers = list(range(N_LAYERS))
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(layers, [sweep[str(i)]["train_acc"] * 100 for i in layers], "o-", color="#9ecae1",
            linewidth=1.5, markersize=5, label="train")
    ax.plot(layers, [sweep[str(i)]["val_acc"] * 100 for i in layers], "s--", color="#6baed6",
            linewidth=1.5, markersize=5, label="val")
    ax.plot(layers, [sweep[str(i)]["test_acc"] * 100 for i in layers], "o-", color="#1f77b4",
            linewidth=2.5, markersize=7, label="test")
    ax.axvline(12, color="#d62728", linewidth=2, linestyle=":", label="layer 12 (chosen)")
    ax.axvline(best, color="#2ca02c", linewidth=1.5, linestyle="--", label=f"best layer by val ({best})")
    ax.set_xlabel("Layer (residual stream output)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Linear probe accuracy across all 26 layers (SST-2, Gemma 2 2B)")
    ax.set_xticks(range(0, N_LAYERS, 2))
    ax.legend(loc="lower center", ncol=2, fontsize=11)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(PLOTS / "layer_sweep_accuracy_nobos.png", dpi=160, bbox_inches="tight")
    plt.close()

    # ---------- 4. direction statistics (raw-space w) ----------
    w = clf.coef_[0].astype(np.float64) / scaler.scale_.astype(np.float64)
    w_unit = w / np.linalg.norm(w)
    sae = SAE.from_pretrained(release="gemma-scope-2b-pt-res-canonical",
                              sae_id="layer_12/width_16k/canonical", device="cpu")
    W_dec = sae.W_dec.detach().cpu().to(torch.float64).numpy()
    W_unit = W_dec / np.linalg.norm(W_dec, axis=1, keepdims=True)
    corrs = np.load(RESULTS / "train_sae_feature_correlations_max.npy")
    ranking = np.argsort(-np.abs(corrs))

    rng = np.random.default_rng(42)
    randv = rng.standard_normal((N_NULL, w.shape[0]))
    randv /= np.linalg.norm(randv, axis=1, keepdims=True)
    null = randv @ w_unit
    cos_feat = float(W_unit[FEATURE_ID] @ w_unit)
    emp = W_unit @ w_unit
    z_rand = float((abs(cos_feat) - abs(null.mean())) / null.std())
    z_emp = float((abs(cos_feat) - np.abs(emp).mean()) / np.abs(emp).std())
    rank = int((np.abs(emp) > abs(cos_feat)).sum()) + 1
    top_frac, rand_frac = [], []
    for K in TOP_K_LIST:
        top_frac.append(captured_fraction(W_dec, ranking[:K], w))
        rand_frac.append(float(np.mean([captured_fraction(
            W_dec, rng.choice(W_dec.shape[0], size=K, replace=False), w) for _ in range(5)])))
    top20 = [{"rank": r + 1, "feature_id": int(f), "correlation": float(corrs[f]),
              "cosine": float(emp[f])} for r, f in enumerate(ranking[:20])]
    stats = {"feature_id": FEATURE_ID, "bos": "excluded", "direction": "raw (coef_/scale_)",
             "n_random_null_vectors": N_NULL, "cosine_feature": cos_feat,
             "random_unit_null": {"mean": float(null.mean()), "std": float(null.std()), "z_score": z_rand,
                                  "percentile": float((np.abs(null) < abs(cos_feat)).mean() * 100)},
             "empirical_sae_null": {"abs_mean": float(np.abs(emp).mean()), "abs_std": float(np.abs(emp).std()),
                                    "z_score": z_emp,
                                    "percentile": float((np.abs(emp) < abs(cos_feat)).mean() * 100)},
             "feature_rank_by_abs_alignment": rank,
             "top20_by_correlation": top20,
             "scaler_sigma": {"min": float(scaler.scale_.min()), "median": float(np.median(scaler.scale_)),
                              "max": float(scaler.scale_.max())},
             "subspace_projection": {"K": TOP_K_LIST, "top_k_fraction": top_frac,
                                     "random_k_fraction": rand_frac}}
    (RESULTS / "direction_stats_nobos.json").write_text(json.dumps(stats, indent=2))
    print(f"\ncos(w, d_{FEATURE_ID}) = {cos_feat:+.4f}  z(random)={z_rand:.1f}  "
          f"z(SAE null)={z_emp:.1f}  alignment rank={rank}")
    for K, a, b in zip(TOP_K_LIST, top_frac, rand_frac):
        print(f"  K={K:4d}  top-K={a:.3f}  random-K={b:.3f}")
    print(f"scaler sigma: min={scaler.scale_.min():.3f} median={np.median(scaler.scale_):.3f} "
          f"max={scaler.scale_.max():.3f}")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(TOP_K_LIST, [v * 100 for v in top_frac], "o-", color="#1f77b4", linewidth=2.5,
            markersize=8, label="top-K SAE features")
    ax.plot(TOP_K_LIST, [v * 100 for v in rand_frac], "s--", color="#9c9489", linewidth=1.5,
            markersize=6, label="random-K features")
    ax.set_xscale("log")
    ax.set_xlabel("K (SAE decoder vectors in the subspace)")
    ax.set_ylabel("% of probe direction ‖w‖ captured")
    ax.set_title("How much of the probe direction lives in the SAE feature subspace")
    ax.legend(loc="upper left", fontsize=12)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(PLOTS / "direction_subspace_projection_nobos.png", dpi=160, bbox_inches="tight")
    plt.close()
    print("Saved all _nobos outputs.")


if __name__ == "__main__":
    main()
