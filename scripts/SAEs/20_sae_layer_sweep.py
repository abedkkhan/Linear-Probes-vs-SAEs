"""
SAE layer sweep: repeat the layer-12 SAE analysis at every layer of Gemma 2 2B,
using the canonical Gemma Scope 16k SAE for each layer.

Phase 1  one forward pass per sentence, all 26 layers captured, BOS dropped,
         token activations written to disk as fp16 (data/SAEs/layer_sweep/).
Phase 2  per layer: encode tokens with that layer's SAE, max-pool
         per sentence, rank features by |Pearson r| on train, then
           - single-feature threshold classifier (threshold chosen on val)
           - top-K logistic regression, K in {1,5,10,20,50,100}
           - no-BOS linear probe at the same layer, raw-space direction w
           - |cos|, z vs random unit vectors, alignment rank, top-20 sign test,
             subspace projection vs random baseline
           - mean L0 per token
         writes results/sae_layer_sweep/layer_{L}.json
Phase 3  aggregate + plots (results/sae_layer_sweep_summary.json,
         results/plots/sae_layer_sweep_*.png)

Every layer is recomputed by default, overwriting results/sae_layer_sweep/. If a run
is interrupted, pass --resume to skip layers whose results already exist.
Layer 12 must reproduce the main results
(top feature 14733, threshold acc 0.744, top-100 acc 0.898, cos -0.229).

Run from project root:
    python scripts/SAEs/20_sae_layer_sweep.py [--resume]
"""

import json
import os
import sys
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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-2-2b"
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
N_LAYERS = 26
SPLITS = ("train", "val", "test")
TOPK = [1, 5, 10, 20, 50, 100]
SUBSPACE_K = [1, 10, 100]
N_NULL = 100_000
CHUNK = 2048

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data"
SWEEP_DATA = DATA / "SAEs" / "layer_sweep"
OUT = ROOT / "results" / "sae_layer_sweep"
PLOTS = ROOT / "results" / "plots"


def column_correlations(X, y):
    y = y.astype(np.float64)
    X = X.astype(np.float64)
    yc = y - y.mean()
    Xc = X - X.mean(axis=0, keepdims=True)
    num = (Xc * yc[:, None]).sum(axis=0)
    den = np.sqrt((Xc ** 2).sum(axis=0) * (yc ** 2).sum())
    r = np.zeros_like(num)
    nz = den > 0
    r[nz] = num[nz] / den[nz]
    return r


def captured_fraction(W_dec, idx, w):
    D = W_dec[idx]
    coef, *_ = np.linalg.lstsq(D.T, w, rcond=None)
    return float(np.linalg.norm(D.T @ coef) / np.linalg.norm(w))


# ---------------------------------------------------------------- phase 1
def extract(device):
    meta_path = SWEEP_DATA / "meta.json"
    if meta_path.exists() and all((SWEEP_DATA / f"tok_L{L}.npy").exists() for L in range(N_LAYERS)):
        print("[phase 1] token activations cached")
        return json.loads(meta_path.read_text())

    SWEEP_DATA.mkdir(parents=True, exist_ok=True)
    rows, split_sizes = [], {}
    for s in SPLITS:
        r = json.loads((DATA / f"{s}_samples.json").read_text())
        split_sizes[s] = len(r)
        rows.extend(r)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    lengths = []
    for r in rows:
        n = len(tok(r["sentence"])["input_ids"])
        lengths.append(n - 1 if n > 1 else n)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    total = int(offsets[-1])
    print(f"[phase 1] {len(rows)} sentences, {total} content tokens")

    mm = [np.lib.format.open_memmap(SWEEP_DATA / f"tok_L{L}.npy", mode="w+",
                                    dtype=np.float16, shape=(total, 2304)) for L in range(N_LAYERS)]
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to(device)
    model.eval()
    captured = {}

    def make_hook(i):
        def hook(_m, _i, output):
            captured[i] = (output[0] if isinstance(output, tuple) else output).detach()
        return hook

    handles = [model.model.layers[i].register_forward_hook(make_hook(i)) for i in range(N_LAYERS)]
    labels = np.empty(len(rows), dtype=np.int64)
    try:
        with torch.no_grad():
            for i, r in enumerate(tqdm(rows, desc="forward")):
                inputs = tok(r["sentence"], return_tensors="pt").to(device)
                assert inputs["input_ids"][0, 0].item() == tok.bos_token_id
                model(**inputs)
                a, b = offsets[i], offsets[i + 1]
                for L in range(N_LAYERS):
                    h = captured[L][0]
                    h = h[1:] if h.shape[0] > 1 else h
                    assert h.shape[0] == b - a
                    mm[L][a:b] = h.to(torch.float16).cpu().numpy()
                labels[i] = r["label"]
    finally:
        for h in handles:
            h.remove()
    for m in mm:
        m.flush()
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    np.save(SWEEP_DATA / "offsets.npy", offsets)
    np.save(SWEEP_DATA / "labels.npy", labels)
    meta = {"split_sizes": split_sizes, "total_tokens": total}
    meta_path.write_text(json.dumps(meta))
    return meta


# ---------------------------------------------------------------- phase 2
def pooled_features(L, sae, device, offsets, n_sent):
    toks = np.load(SWEEP_DATA / f"tok_L{L}.npy", mmap_mode="r")
    seg = torch.from_numpy(np.repeat(np.arange(n_sent), np.diff(offsets)))
    g = torch.zeros((n_sent, sae.cfg.d_sae), dtype=torch.float32)
    l0_sum, n_tok = 0.0, 0
    with torch.no_grad():
        for a in range(0, toks.shape[0], CHUNK):
            x = torch.from_numpy(np.asarray(toks[a:a + CHUNK], dtype=np.float32)).to(device)
            f = sae.encode(x)
            l0_sum += float((f != 0).sum().item())
            n_tok += f.shape[0]
            f = f.to(torch.float16).to(torch.float32).cpu()
            idx = seg[a:a + f.shape[0]].unsqueeze(1).expand(-1, f.shape[1])
            g.scatter_reduce_(0, idx, f, reduce="amax", include_self=True)
    return g.numpy(), l0_sum / n_tok


def single_feature_classifier(fv_val, y_val, fv_test, y_test, corr):
    # positive-correlated feature -> predict positive when above threshold
    sign = 1.0 if corr > 0 else -1.0
    cands = np.unique(fv_val)
    best_thr, best_acc = None, -1.0
    for thr in cands:
        pred = (fv_val > thr).astype(int) if sign > 0 else (fv_val <= thr).astype(int)
        acc = accuracy_score(y_val, pred)
        if acc > best_acc:
            best_acc, best_thr = acc, float(thr)
    pred = (fv_test > best_thr).astype(int) if sign > 0 else (fv_test <= best_thr).astype(int)
    score = sign * fv_test
    return {"threshold": best_thr, "val_acc": float(best_acc),
            "test_acc": float(accuracy_score(y_test, pred)),
            "test_f1": float(f1_score(y_test, pred)),
            "test_auc": float(roc_auc_score(y_test, score))}


RESUME = "--resume" in sys.argv   # skip layers whose results already exist

def analyse_layer(L, device, meta, offsets, labels, randv):
    out_path = OUT / f"layer_{L}.json"
    if RESUME and out_path.exists():
        return json.loads(out_path.read_text())
    t0 = time.time()
    n_tr, n_va = meta["split_sizes"]["train"], meta["split_sizes"]["val"]
    n = len(labels)
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=f"layer_{L}/width_16k/canonical", device=str(device))
    sae.eval()
    g, l0 = pooled_features(L, sae, device, offsets, n)
    W_dec = sae.W_dec.detach().cpu().to(torch.float64).numpy()
    del sae
    if device.type == "mps":
        torch.mps.empty_cache()

    ytr, yva, yte = labels[:n_tr], labels[n_tr:n_tr + n_va], labels[n_tr + n_va:]
    gtr, gva, gte = g[:n_tr], g[n_tr:n_tr + n_va], g[n_tr + n_va:]
    corrs = column_correlations(gtr, ytr)
    ranking = np.argsort(-np.abs(corrs))
    top = int(ranking[0])

    single = single_feature_classifier(gva[:, top], yva, gte[:, top], yte, corrs[top])

    topk = {}
    for K in TOPK:
        idx = ranking[:K]
        sc = StandardScaler()
        Xtr = sc.fit_transform(gtr[:, idx])
        clf = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
        clf.fit(Xtr, ytr)
        pv = clf.predict(sc.transform(gva[:, idx]))
        pt = clf.predict(sc.transform(gte[:, idx]))
        topk[str(K)] = {"val_acc": float(accuracy_score(yva, pv)), "test_acc": float(accuracy_score(yte, pt))}
    best_k = max(TOPK, key=lambda k: topk[str(k)]["val_acc"])

    # no-BOS linear probe at this layer (same recipe as 06_probe_pipeline_nobos.py)
    Xtr_p = np.load(DATA / "linear_probes" / "all_layers_nobos_train.npy", mmap_mode="r")[:, L]
    Xte_p = np.load(DATA / "linear_probes" / "all_layers_nobos_test.npy", mmap_mode="r")[:, L]
    psc = StandardScaler()
    pclf = LogisticRegression(C=1.0, max_iter=1000, random_state=42, penalty="l2", solver="lbfgs")
    pclf.fit(psc.fit_transform(np.asarray(Xtr_p)), ytr)
    probe_acc = float(accuracy_score(yte, pclf.predict(psc.transform(np.asarray(Xte_p)))))
    w = pclf.coef_[0].astype(np.float64) / psc.scale_.astype(np.float64)
    w_unit = w / np.linalg.norm(w)

    W_unit = W_dec / np.linalg.norm(W_dec, axis=1, keepdims=True)
    emp = W_unit @ w_unit
    cos_top = float(emp[top])
    null = randv @ w_unit.astype(np.float32)
    z_rand = float((abs(cos_top) - abs(float(null.mean()))) / float(null.std()))
    z_emp = float((abs(cos_top) - np.abs(emp).mean()) / np.abs(emp).std())
    rank = int((np.abs(emp) > abs(cos_top)).sum()) + 1
    t20 = ranking[:20]
    sign_agree = int(sum((corrs[f] > 0) == (emp[f] > 0) for f in t20))
    rng = np.random.default_rng(1000 + L)
    sub = {}
    for K in SUBSPACE_K:
        rand = float(np.mean([captured_fraction(W_dec, rng.choice(W_dec.shape[0], K, replace=False), w)
                              for _ in range(5)]))
        sub[str(K)] = {"top": captured_fraction(W_dec, ranking[:K], w), "random": rand}

    res = {"layer": L, "mean_l0": l0, "top_feature": top, "top_corr": float(corrs[top]),
           "single_feature": single, "topk": topk, "best_k_by_val": best_k,
           "best_k_test_acc": topk[str(best_k)]["test_acc"], "probe_test_acc": probe_acc,
           "cos_top": cos_top, "z_random": z_rand, "z_sae_null": z_emp, "alignment_rank": rank,
           "sign_agreement_top20": sign_agree, "subspace": sub, "seconds": time.time() - t0}
    OUT.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2))
    print(f"[L{L:2d}] top={top:5d} r={corrs[top]:+.3f}  single={single['test_acc']:.3f}  "
          f"top100={topk['100']['test_acc']:.3f}  bestK={best_k}->{res['best_k_test_acc']:.3f}  "
          f"probe={probe_acc:.3f}  |cos|={abs(cos_top):.3f} z={z_rand:.1f} rank={rank} "
          f"sign={sign_agree}/20  rho100={sub['100']['top']:.3f}/{sub['100']['random']:.3f}  "
          f"L0={l0:.0f}  ({res['seconds']:.0f}s)", flush=True)
    return res


# ---------------------------------------------------------------- phase 3
def plots(all_res):
    sns.set_theme(style="whitegrid", context="talk")
    Ls = [r["layer"] for r in all_res]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(Ls, [r["probe_test_acc"] * 100 for r in all_res], "o-", color="#d62728", lw=2.5, label="Linear probe")
    ax.plot(Ls, [r["topk"]["100"]["test_acc"] * 100 for r in all_res], "o-", color="#1f77b4", lw=2.5,
            label="SAE top-100 classifier")
    ax.plot(Ls, [r["best_k_test_acc"] * 100 for r in all_res], "s--", color="#1f77b4", alpha=0.6, lw=1.5,
            label="SAE top-K classifier (K chosen on val)")
    ax.plot(Ls, [r["single_feature"]["test_acc"] * 100 for r in all_res], "^-", color="#ff7f0e", lw=1.5,
            label="SAE single feature (threshold)")
    ax.axvline(12, color="grey", ls=":", lw=1.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Probe and SAE classification accuracy across layers")
    ax.set_xticks(range(0, N_LAYERS, 2))
    ax.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    plt.savefig(PLOTS / "sae_layer_sweep_accuracy.png", dpi=160, bbox_inches="tight")
    plt.close()

    c = [abs(r["cos_top"]) for r in all_res]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(Ls, c, "-", color="#1f77b4", lw=2)
    r1 = [(l, v) for l, v, r in zip(Ls, c, all_res) if r["alignment_rank"] == 1]
    rn = [(l, v) for l, v, r in zip(Ls, c, all_res) if r["alignment_rank"] != 1]
    ax.plot([p[0] for p in r1], [p[1] for p in r1], "o", color="#1f77b4", ms=10,
            label="top feature is the most aligned of all 16,384")
    ax.plot([p[0] for p in rn], [p[1] for p in rn], "o", mfc="white", mec="#1f77b4", mew=2, ms=10,
            label="top feature not the most aligned")
    ax.axhline(3 / np.sqrt(2304), color="grey", ls="--", lw=1.5, label="3 s.d. of random-vector null")
    ax.axvline(12, color="grey", ls=":", lw=1.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("|cos(w, d_top)|")
    ax.set_ylim(0, 0.35)
    ax.set_xticks(range(0, N_LAYERS, 2))
    ax.set_title("Alignment of the top SAE feature with the probe direction")
    ax.legend(loc="upper right", fontsize=11)
    plt.tight_layout()
    plt.savefig(PLOTS / "sae_layer_sweep_alignment.png", dpi=160, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(Ls, [r["subspace"]["100"]["top"] * 100 for r in all_res], "o-", color="#1f77b4", lw=2.5,
            label="top-100 SAE features")
    ax.plot(Ls, [r["subspace"]["100"]["random"] * 100 for r in all_res], "s--", color="#9c9489", lw=1.5,
            label="100 random features")
    ax.axvline(12, color="grey", ls=":", lw=1.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("% of probe direction captured")
    ax.set_title("Probe direction captured by the span of the top-100 SAE decoders")
    ax.set_xticks(range(0, N_LAYERS, 2))
    ax.legend(loc="upper right", fontsize=11)
    plt.tight_layout()
    plt.savefig(PLOTS / "sae_layer_sweep_subspace.png", dpi=160, bbox_inches="tight")
    plt.close()


def main():
    load_dotenv(ROOT / ".env")
    login(token=os.environ["HUGGING_FACE_TOKEN"], add_to_git_credential=False)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    meta = extract(device)
    offsets = np.load(SWEEP_DATA / "offsets.npy")
    labels = np.load(SWEEP_DATA / "labels.npy")
    ref = np.concatenate([np.load(DATA / "SAEs" / f"{s}_labels.npy") for s in SPLITS])
    assert np.array_equal(labels, ref), "label order mismatch"

    rng = np.random.default_rng(42)
    randv = rng.standard_normal((N_NULL, 2304)).astype(np.float32)
    randv /= np.linalg.norm(randv, axis=1, keepdims=True)

    order = [12] + [L for L in range(N_LAYERS) if L != 12]   # validate on layer 12 first
    all_res = {}
    for L in order:
        all_res[L] = analyse_layer(L, device, meta, offsets, labels, randv)
        if L == 12:
            r = all_res[12]
            print(f"[check] layer 12: top={r['top_feature']} (expect 14733)  "
                  f"single={r['single_feature']['test_acc']:.3f} (expect 0.744)  "
                  f"top100={r['topk']['100']['test_acc']:.3f} (expect 0.898)  "
                  f"cos={r['cos_top']:+.3f} (expect -0.229)", flush=True)

    res_list = [all_res[L] for L in range(N_LAYERS)]
    (ROOT / "results" / "sae_layer_sweep_summary.json").write_text(json.dumps(res_list, indent=2))
    plots(res_list)
    print("Done. Summary: results/sae_layer_sweep_summary.json", flush=True)


if __name__ == "__main__":
    main()
