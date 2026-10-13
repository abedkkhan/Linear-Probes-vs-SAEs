# Beyond Accuracy: Do Linear Probes and Sparse Autoencoders Recover the Same Representation?

Code and results for a head-to-head comparison of **linear probes** and **sparse autoencoders (SAEs)** on the same model, layer, and task: binary sentiment (SST-2) in the residual stream of **Gemma 2 2B**, using the pretrained **Gemma Scope** SAEs. The two methods are compared on classification accuracy and on the geometry of the directions they identify.

## Main results (layer 12, 500 test sentences)

| | Accuracy | F1 | AUC-ROC |
|---|---|---|---|
| Linear probe | 0.922 | 0.931 | 0.973 |
| SAE, top-100 features | 0.898 | 0.910 | 0.958 |
| SAE, single feature (#14,733) | 0.744 | 0.792 | 0.808 |

- The probe vs top-100 SAE gap is **not significant** (exact McNemar, p = 0.07).
- The top SAE feature is the **most aligned of all 16,384 features** with the probe direction (|cos| = 0.23, z = 11.0), and the sign of alignment matches feature polarity for all top-20 features.
- The alignment is **partial**: the top 100 SAE decoders span 44% of the probe direction.
- Across all 26 layers the directional agreement holds, while the accuracy ranking depends on depth.

All numbers and figures are in [`results/`](results/) and [`results/plots/`](results/plots/).

## Setup

Requires Python 3.12 and about 10 GB of free memory (Apple-silicon MPS, CUDA, or CPU).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # then add your Hugging Face token
```

Gemma 2 is a gated model: accept its licence at <https://huggingface.co/google/gemma-2-2b> before running.

## Reproduce

```bash
./reproduce.sh
```

This runs the full pipeline in order and regenerates everything in `results/`. Runtime is about 1 hour on an Apple M-series machine, most of it model forward passes and the layer sweep. Large intermediates (activations, SAE features, trained probes) are written to `data/` and `models/`, which are not tracked.

All randomness uses seed 42. Results should match `results/` exactly on the same hardware; small numerical differences can occur across devices.

## Where each result comes from

| Paper | Script(s) | Output |
|---|---|---|
| Data split (Sec. 3.2) | `Linear_ptobes/01_prepare_dataset.py` | `data/*_samples.json` |
| Probe, layer sweep, direction statistics (Sec. 4.1) | `Linear_ptobes/06_probe_pipeline_nobos.py` | `probe_metrics_nobos.json`, `layer_sweep_metrics_nobos.json`, `direction_stats_nobos.json` |
| SAE classifiers (Table 1, Fig. top-K) | `SAEs/05`–`11`, `SAEs/19` | `sae_classifier_metrics.json`, `sae_topk_metrics.json` |
| Feature 14,733 selectivity and examples (Sec. 4.1.4, 4.2) | `SAEs/13`, `SAEs/17`, `SAEs/23` | `feature_14733_interpretability.json`, `sae_sparsity_feature_stats.json` |
| Probe vs SAE errors (Sec. 4.2) | `Linear_ptobes/07_qualitative_nobos.py` | `qualitative_nobos.json` |
| Significance tests | `SAEs/21_significance_tests.py` | `significance_tests_nobos.json` |
| Phrase-overlap robustness check (Limitations) | `SAEs/24_overlap_robustness.py` | `overlap_robustness.json` |
| Confusion matrices (App. C) | `SAEs/22_confusion_matrices.py` | `confusion_matrices_nobos.json` |
| Effect of corrections (App. D) | `Linear_ptobes/03`, `SAEs/12`, `16`, `18` | `direction_stats.json`, `direction_stats_raw.json` |
| SAE layer sweep (App. E) | `SAEs/20_sae_layer_sweep.py` | `sae_layer_sweep_summary.json` |

Two details matter for a fair comparison: the beginning-of-sequence token is excluded for both methods, and the probe direction is mapped back from standardised to raw activation space (`w = coef / scale`) before it is compared with SAE decoder vectors.

## Repository layout

```
scripts/Linear_ptobes/   data preparation, activation extraction, linear probes
scripts/SAEs/            SAE encoding, feature selection, comparison, figures
results/                 metrics (JSON) and figures reproduced by reproduce.sh
reproduce.sh             runs the full pipeline in order
```

Not required for the paper (sanity checks or superseded analyses): `Linear_ptobes/00`, `02a`, `02b`, `04`, `05` and `SAEs/05a`, `05b`, `14`, `15`.

## Authors

Aabid Karim, Nargiz Sultanova, Savin Chand (Federation University Australia).
