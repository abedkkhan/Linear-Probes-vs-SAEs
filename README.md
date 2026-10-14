# Beyond Accuracy: Do Linear Probes and Sparse Autoencoders Recover the Same Representation?

Code and results for a head-to-head comparison of **linear probes** and **sparse autoencoders (SAEs)** on the same model, layer, and task: binary sentiment (SST-2) in the residual stream of **Gemma 2 2B**, using the pretrained **Gemma Scope** SAEs. The two methods are compared on classification accuracy and on the geometry of the directions they identify.

## Main results (layer 12, 500 test sentences)

| | Accuracy | F1 | AUC-ROC |
|---|---|---|---|
| Linear probe | 0.922 | 0.931 | 0.973 |
| SAE, top-100 features | 0.898 | 0.910 | 0.958 |
| SAE, single feature (#14,733) | 0.744 | 0.792 | 0.808 |

- The probe is **more accurate than every SAE classifier** at layer 12. K = 100 is the best value on the test set, and for it the gap is not significant (exact McNemar, p = 0.07); for K selected on the validation set (K = 50, 87.6%) it is (p < 0.001).
- The top SAE feature is the **most aligned of all 16,384 features** with the probe direction (|cos| = 0.23, z = 11.0), and the sign of alignment matches feature polarity for all top-20 features.
- The alignment is **partial**: the top 100 SAE decoder vectors span 44% of the probe direction.
- The directional agreement holds at **25 of the 26 layers**, while the accuracy ranking depends on depth.

All numbers and figures are in [`results/`](results/) and [`results/plots/`](results/plots/).

## Setup

Requires Python 3.12, about 10 GB of free memory (Apple-silicon MPS, CUDA, or CPU) and about 25 GB of free disk: 17 GB for the model and SAE downloads, cached by Hugging Face on the first run, and 7 GB for intermediate files.

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

This runs the full pipeline in order and regenerates every number, table and figure in the paper. Once the model and SAEs are downloaded, it takes about 25 minutes on an Apple M-series machine, most of it model forward passes. Large intermediates (activations, SAE features, trained probes) are written to `data/` and `models/`, which are not tracked. The comparison at every layer (Appendix E) recomputes all 26 layers; if it is interrupted, continue it with `python scripts/SAEs/20_sae_layer_sweep.py --resume`.

All randomness uses seed 42. A from-scratch run of `reproduce.sh` in a fresh clone of this repository (Apple M-series, MPS) reproduced every committed result: the data split is byte-identical, every figure is identical, and all numeric values match, apart from differences of about 1e-16 from floating-point rounding. Small numerical differences can occur on other devices.

## Where each result comes from

| Paper | Script(s) | Output |
|---|---|---|
| Data split (Sec. 2.2) | `Linear_ptobes/01_prepare_dataset.py` | `data/*_samples.json` |
| Linear probe, layer sensitivity, direction statistics (Sec. 3.1) | `Linear_ptobes/06_probe_pipeline_nobos.py` | `probe_metrics_nobos.json`, `layer_sweep_metrics_nobos.json`, `direction_stats_nobos.json` |
| SAE classifiers (Sec. 3.1.1, Table 1) | `SAEs/05`–`11`, `SAEs/19` | `sae_classifier_metrics.json`, `sae_topk_metrics.json` |
| Feature 14,733 selectivity and examples (Sec. 3.1.4, 3.2) | `SAEs/13`, `SAEs/17`, `SAEs/23` | `feature_14733_interpretability.json`, `sae_sparsity_feature_stats.json` |
| Probe vs SAE errors (Sec. 3.2) | `Linear_ptobes/07_qualitative_nobos.py` | `qualitative_nobos.json` |
| Significance tests (Sec. 3.1.1) | `SAEs/21_significance_tests.py` | `significance_tests_nobos.json` |
| Phrase-overlap robustness check (Sec. 4.3) | `SAEs/24_overlap_robustness.py` | `overlap_robustness.json` |
| Confusion matrices (App. C) | `SAEs/22_confusion_matrices.py` | `confusion_matrices_nobos.json` |
| Effect of corrections (App. D) | `Linear_ptobes/03`, `SAEs/12`, `16`, `18` | `direction_stats.json`, `direction_stats_raw.json` |
| Comparison at every layer (App. E) | `SAEs/20_sae_layer_sweep.py` | `sae_layer_sweep_summary.json` |

The figures in the paper are in `results/plots/`: the files ending in `_nobos`, `feature_14733_*.png` and `sae_layer_sweep_*.png`.

Two details matter for a fair comparison: the beginning-of-sequence token is excluded for both methods, and the probe direction is mapped back from standardised to raw activation space (`w = coef / scale`) before it is compared with SAE decoder vectors.

## Repository layout

```
scripts/Linear_ptobes/   data preparation, activation extraction, linear probes
scripts/SAEs/            SAE encoding, feature selection, comparison, figures
results/                 metrics (JSON) and figures reproduced by reproduce.sh
reproduce.sh             runs the full pipeline in order
```

Not required for the paper (sanity checks or superseded analyses): `Linear_ptobes/00`, `02a`, `02b`, `04`, `05` and `SAEs/05a`, `05b`, `14`, `15`. Some of their outputs are kept in `results/` for reference and are not regenerated by `reproduce.sh`, for example `layer_sweep_metrics.json`, `probe_qualitative.json` and `top_weight_dimensions.json`, which come from an earlier analysis that included the beginning-of-sequence token.

## Authors

Aabid Karim, Nargiz Sultanova, Savin Chand (Federation University Australia).
