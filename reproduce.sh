#!/usr/bin/env bash
# Reproduce every number, table and figure in the paper.
# Run from the repository root after `pip install -r requirements.txt` and creating .env.
set -euo pipefail
LP=scripts/Linear_ptobes
SAE=scripts/SAEs
run() { echo -e "\n==> $1"; python "$1"; }

# 1. Data: fixed 2,000 / 500 / 500 SST-2 split (seed 42)
run $LP/01_prepare_dataset.py

# 2. SAE pipeline at layer 12 (BOS excluded)
run $SAE/05_extract_token_activations.py
run $SAE/06_encode_per_token_sae.py
run $SAE/07_pool_sae_features.py
run $SAE/08_find_sentiment_feature.py
run $SAE/09_find_cutoff.py
run $SAE/10_evaluate_sae_classifier.py
run $SAE/11_topk_sae_probe.py
run $SAE/17_feature_interpretability.py

# 3. Linear probe as in the paper: BOS excluded, layer 12 + sweep over all 26 layers,
#    raw-space probe direction vs SAE decoders
run $LP/02_extract_activations.py        # also writes the label files used below
run $LP/06_probe_pipeline_nobos.py
run $LP/07_qualitative_nobos.py

# 4. Uncorrected baselines reported in Appendix D (BOS included, standardised direction)
run $LP/03_train_linear_probe.py
run $SAE/12_direction_comparison.py
run $SAE/16_direction_stats.py
run $SAE/18_direction_stats_raw.py
run $SAE/13_make_plots.py                # includes the feature-14733 activation histogram

# 5. Paper figures and statistics
run $SAE/19_plot_topk_vs_probe_nobos.py
run $SAE/21_significance_tests.py
run $SAE/22_confusion_matrices.py
run $SAE/23_sae_sparsity_and_feature_stats.py

# 6. SAE layer sweep (Appendix E); resumable, about 30 min on Apple silicon
run $SAE/20_sae_layer_sweep.py

echo -e "\nDone. Results in results/, figures in results/plots/."
