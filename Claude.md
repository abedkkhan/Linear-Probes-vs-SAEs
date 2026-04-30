# Linear Probes vs Sparse Autoencoders: Initial Comparison Experiment

## Project Overview

This is a Final Year Project (FYP) comparing two methods for extracting interpretable internal representations from language models:

1. **Linear Probes** - Logistic regression classifiers trained on frozen model activations
2. **Sparse Autoencoders (SAEs)** - Pretrained autoencoders that decompose activations into sparse, interpretable features

The goal is to empirically compare these methods on sentiment analysis to understand which finds better representations of sentiment in a pretrained language model.

---

## Research Hypothesis

**Hypothesis**: Both linear probes and sparse autoencoders should identify similar directions in activation space corresponding to sentiment, but they may differ in:
- Predictive accuracy on classification tasks
- Geometric interpretation (cosine similarity between found directions)
- Interpretability of the features they identify

**Sub-questions**:
1. Do both methods find sentiment as a linear direction in the model's activation space?
2. Which method achieves higher classification accuracy on held-out test data?
3. Do the directions found by each method align (high cosine similarity)?

---

## Research Objectives

1. **Implement linear probing** on a pretrained LLM for sentiment classification
2. **Apply pretrained SAEs** (from Gemma Scope) to the same activations
3. **Compare both methods empirically** using standard metrics
4. **Document findings** with clear visualizations

---

## Initial Experiment Specifications

This is a **minimum viable experiment** to validate the methodology. We will scale up later.

### Model
- **Name**: Gemma 2 2B
- **Hugging Face ID**: `google/gemma-2-2b`
- **Why this model**: Has pretrained SAEs available via Gemma Scope, allowing direct comparison without training SAEs from scratch
- **Frozen**: We do NOT train or fine-tune the model

### Dataset
- **Name**: Stanford Sentiment Treebank (SST-2)
- **Hugging Face ID**: `stanfordnlp/sst2`
- **Task**: Binary sentiment classification (positive vs negative)
- **Initial subset for fast iteration**:
  - 2,000 training samples
  - 500 validation samples
  - 500 test samples
- **Reason for subset**: Keep initial experiment fast; scale up after validation

### Layer to Probe
- **Layer**: 12 (residual stream)
- **Why this layer**: Middle layer where semantic concepts (like sentiment) typically emerge
- **Pooling strategy**: Mean pooling across all token activations to get one vector per sentence
- **Output dimension**: 2,304 (Gemma 2 2B residual stream size)

### Linear Probe
- **Classifier**: Logistic Regression with L2 regularization
- **Library**: scikit-learn (`sklearn.linear_model.LogisticRegression`)
- **Hyperparameters**:
  - `C=1.0` (regularization strength)
  - `max_iter=1000`
  - `random_state=42` (for reproducibility)
- **Preprocessing**: StandardScaler to normalize features
- **Why logistic regression**: Standard in linear probing literature, interpretable weights ARE the concept direction

### Sparse Autoencoder
- **Source**: Gemma Scope (Google DeepMind)
- **Hugging Face ID**: `google/gemma-scope-2b-pt-res`
- **Specific SAE**: Layer 12, width 16K (smallest version for fast iteration)
- **Library**: SAELens (`pip install sae-lens`)
- **Architecture**: JumpReLU SAE (pretrained, no training needed)

---

## Implementation Plan

### Phase 1: Environment Setup

**Required Libraries**:
```
torch
transformers
datasets
scikit-learn
sae-lens
matplotlib
seaborn
pandas
numpy
```

**Verify**:
- Gemma 2 2B loads successfully
- SAE loads from Hugging Face
- GPU available (or CPU works for small subset)

### Phase 2: Data Pipeline

1. Load SST-2 dataset from Hugging Face
2. Take subsets: 2000 train, 500 val, 500 test
3. Tokenize sentences (use Gemma's tokenizer)
4. Create batches for efficient processing

### Phase 3: Activation Extraction

For each sentence:
1. Tokenize and feed through frozen Gemma 2 2B
2. Hook into Layer 12 residual stream
3. Extract activations for all tokens
4. Apply mean pooling → single vector (2,304 dims)
5. Save as numpy array with corresponding labels

**Output**: 
- `train_activations.npy` (2000 × 2304)
- `train_labels.npy` (2000,)
- Same for validation and test sets

### Phase 4: Linear Probe Training

1. Standardize activations (StandardScaler fit on train, transform all)
2. Train LogisticRegression on training set
3. Evaluate on test set:
   - Accuracy
   - F1 score
   - AUC-ROC
4. Save the trained probe and the weight vector (sentiment direction)

### Phase 5: SAE Analysis

1. Load Gemma Scope SAE (layer 12, 16K)
2. Pass training activations through SAE.encode() → sparse features
3. For each of the 16K features, compute correlation with sentiment label
4. Identify top-K features most correlated with sentiment
5. Use top SAE feature(s) as a classifier:
   - Treat feature activation as a sentiment score
   - Set threshold, classify, evaluate accuracy

### Phase 6: Comparison Analysis

**Quantitative Comparisons**:
1. Classification accuracy: probe vs SAE-based classifier
2. Cosine similarity between:
   - Linear probe weight vector
   - Top SAE feature's decoder vector
3. Number of "useful" SAE features for sentiment (correlation > threshold)

**Visualizations**:
1. Accuracy comparison bar chart
2. Top 10 SAE features by sentiment correlation
3. Activation patterns: top-activating sentences for top SAE feature
4. Histogram of feature activations for positive vs negative samples

### Phase 7: Documentation

Generate:
- Results table (accuracy, F1, AUC for each method)
- Plots saved as PNG files
- Summary markdown report

---

## Expected Outputs

### Files to Generate
```
project/
├── data/
│   ├── train_activations.npy
│   ├── train_labels.npy
│   ├── val_activations.npy
│   ├── val_labels.npy
│   ├── test_activations.npy
│   └── test_labels.npy
├── models/
│   ├── linear_probe.pkl
│   └── scaler.pkl
├── results/
│   ├── metrics.json
│   ├── probe_direction.npy
│   ├── top_sae_features.json
│   └── plots/
│       ├── accuracy_comparison.png
│       ├── sae_feature_correlations.png
│       ├── top_feature_activations.png
│       └── direction_similarity.png
├── scripts/
│   ├── 01_extract_activations.py
│   ├── 02_train_linear_probe.py
│   ├── 03_analyze_sae.py
│   ├── 04_compare_methods.py
│   └── 05_generate_plots.py
└── report.md
```

---

## Key Code Structure Recommendations

### Modular Design
- Each phase as a separate script that can be run independently
- Save intermediate results (activations, etc.) to disk
- This way, if one phase fails, you don't have to re-extract activations (the slowest step)

### Reproducibility
- Fix random seeds (42)
- Save all hyperparameters in a config file
- Log the exact dataset subset used (sample IDs)

### Efficiency
- Batch processing for activation extraction
- Use GPU if available (huge speedup)
- Cache activations to disk—extracting them is the slowest step

---

## Success Criteria

This initial experiment is successful if:

1. ✓ Linear probe achieves > 75% accuracy on SST-2 test set
2. ✓ At least one SAE feature is identified that correlates with sentiment
3. ✓ Both methods can be quantitatively compared
4. ✓ Clean visualizations are produced
5. ✓ Results are reproducible (re-running gives same numbers)

If linear probe achieves > 85%, that's even better. SAE features are more variable—getting a clear sentiment-aligned feature is the goal.

---

## Important Notes for Claude Code

### Things to Be Careful About

1. **Memory**: Gemma 2 2B is ~5GB in float16, ~10GB in float32. Use float16 if possible.

2. **Tokenization**: Gemma uses a specific tokenizer. Don't mix with other models' tokenizers.

3. **Layer indexing**: Layer 12 means index 12 (zero-indexed). Verify by checking the model architecture.

4. **Hook placement**: Make sure activation hooks are on the residual stream OUTPUT of layer 12, not the input.

5. **Mean pooling**: Pool over the sequence dimension, not the batch dimension.

6. **SAE usage**: Use the encoder forward pass: `features = sae.encode(activations)`.

7. **Sparsity check**: Verify that SAE features are actually sparse (most should be zero).

### Things NOT to Do

1. ❌ Don't fine-tune the model
2. ❌ Don't train a new SAE (use Gemma Scope's pretrained ones)
3. ❌ Don't use the full SST-2 dataset for initial experiment (too slow)
4. ❌ Don't skip standardization for linear probe
5. ❌ Don't use complex pooling strategies (mean pooling is fine)

### Things to Verify

1. Activation shapes are correct: `(batch_size, seq_len, hidden_dim)` before pooling, `(batch_size, hidden_dim)` after
2. Labels are aligned with activations
3. Train/test split has no overlap
4. SAE features are sparse (most should be exactly zero or near-zero)

---

## Timeline

- **Day 1**: Environment setup, verify model and SAE load
- **Day 2**: Activation extraction script
- **Day 3**: Linear probe training and evaluation
- **Day 4**: SAE analysis
- **Day 5**: Comparison and visualization
- **Day 6**: Documentation and refinement

**Goal**: Have results ready to present in a Week 8 progress meeting.

---

## Future Extensions (NOT for this initial experiment)

After validating the methodology, future work will include:
- Test multiple layers (find best for sentiment)
- Use full SST-2 + IMDb datasets
- Larger SAE variants (65K, 131K features)
- Different pooling strategies
- Statistical significance testing
- Error analysis (where do methods disagree?)

But for now: **focus on getting the basic pipeline working with clean results**.

---

## References

Key papers this work builds on:
1. Alain & Bengio (2017) - "Understanding intermediate layers using linear classifier probes"
2. Park et al. (2024) - "The Linear Representation Hypothesis and the Geometry of Large Language Models"
3. Cunningham et al. (2023) - "Sparse Autoencoders Find Highly Interpretable Features in Language Models"
4. Lieberum et al. (2024) - "Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2"

---

## Contact

**Student**: Aabid Karim (30468176)
**Project**: Comparing Linear Probes vs Sparse Autoencoders for Internal Knowledge Representation in Language Models