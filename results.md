# Initial Results — Linear Probe on Gemma 2 2B, Layer 12

This document records the first set of results from the project: a linear probe trained on Gemma 2 2B's layer-12 residual stream activations to classify SST-2 sentences as positive or negative.

---

## 1. Headline numbers

| Split | n    |Accuracy| F1     | AUC-ROC |
|-------|------|--------|--------|--------|
| Train | 2000 | 1.0000 | 1.0000 | 1.0000 |
| Val   |  500 | 0.9140 | 0.9171 | 0.9688 |
| Test  |  500 | 0.9200 | 0.9293 | 0.9728 |

**What this means in plain language.** A logistic regression — a model with just one weight per activation dimension and no non-linearities — can correctly classify the sentiment of 92% of unseen movie-review snippets, simply by looking at Gemma's internal state at layer 12. It scored 100% on the training data (it has enough capacity to memorise 2000 examples) and ~92% on examples it has never seen, with only an 8-point gap between train and test. That gap is healthy: the probe has learned a *general* pattern, not memorised noise.

**Why this matters.** This is the empirical evidence for the **Linear Representation Hypothesis** at this specific layer of this specific model. If sentiment is recoverable by a linear classifier, it means sentiment is encoded as a single straight-line direction in Gemma's 2304-dimensional activation space — you can walk along that direction in the space and your sentence becomes more or less positive in a predictable way. The probe's weight vector *is* that direction.

---

## 2. Where it succeeds — qualitative examples

The five test sentences the probe was most confident about (and got right) on each side:

**Most-confident correct positives** (logit > +22):
- *"an enthralling aesthetic experience"*
- *"a fascinating, compelling story"*
- *"that, with humor, warmth, and intelligence, captures a life interestingly lived"*
- *"a brilliant, absurd collection"*
- *"powerful and astonishingly vivid"*

**Most-confident correct negatives** (logit < −25):
- *"pathetic junk"*
- *"the insipid script"*
- *"run out of clever ideas and visual gags about halfway through"*
- *"without any redeeming value whatsoever"*
- *"like being stuck in a dark pit having a nightmare about bad cinema"*

**What this tells us.** Both lists contain sentences with *explicit, unambiguous* sentiment vocabulary — words like "enthralling", "brilliant", "pathetic", "insipid". The probe is most confident exactly where humans would be: on sentences whose sentiment is on the surface. The very large logit magnitudes (up to ±29) tell us these examples sit far out along the sentiment direction in activation space — Gemma represents them as strongly positive or strongly negative internally, and the probe just reads that off.

---

## 3. Where it fails — and what the failures reveal

The five test sentences the probe got most confidently *wrong*:

| Logit | True | Predicted | Sentence |
|-------|------|-----------|----------|
| +8.08 | neg | pos | *"you can taste it, but there's no fizz"* |
| −7.61 | pos | neg | *"for cryin'"* |
| +7.60 | neg | pos | *"a rude black comedy about the catalytic effect a holy fool has upon those around him..."* |
| +7.09 | neg | pos | *"genial-rogue shtick"* |
| +7.03 | neg | pos | *"( or role, or edit, or score, or anything, really )"* |

**What this tells us.** The mistakes are not random. They cluster around three patterns:

1. **Metaphorical negativity.** *"there's no fizz"* uses positive surface words ("taste", "fizz") to express disappointment. The negativity is structural, not lexical.
2. **Description rather than evaluation.** *"a rude black comedy about..."* describes a film's content; it doesn't explicitly praise or condemn it.
3. **Fragmentary phrases.** *"for cryin'"* and *"genial-rogue shtick"* are too short and too context-dependent to carry clear sentiment on their own.

**The conclusion.** The layer-12 sentiment direction is good at picking up *lexical-level* sentiment — it recognises the words "enthralling" and "pathetic". It struggles with sentiment that requires structural reasoning ("there's no X"), pure description, or fragmented context. This is consistent with what we'd expect from a single linear direction: it can capture a strong but relatively shallow signal.

---

## 4. What we learned about the model

Three concrete claims supported by the experiment:

1. **Gemma 2 2B does internally represent sentiment.** Even though the model was never trained on labelled sentiment data, it has learned to track sentiment as part of next-token prediction — because to predict the next word in "the movie was utterly...", you have to know which way the review is going.

2. **At layer 12 (the middle of the model), this representation is approximately linear.** A simple dot product against a fixed direction recovers the label 92% of the time on held-out data.

3. **The signal is distributed, not localised.** The probe's largest weights are around ±0.34 — substantial, but no single dimension dominates. The top 5 positive-weight and negative-weight dimensions each contribute meaningfully, and the overall weight vector has a moderate L2 norm of 4.60. Sentiment is encoded by *many* residual-stream dimensions working together, not by a single "sentiment neuron".

This last point is exactly what motivates Phase 5 of the project. A linear probe gives us a direction, but the direction is an opaque mixture of 2304 individual dimensions — we can't point to one dimension and say "this means sadness". A sparse autoencoder, in principle, decomposes the same activation space into a much larger dictionary of features where each individual feature is more interpretable. The next phase will test whether the SAE finds a *single feature* that aligns with the probe's direction.

---

## 5. What we learned about the dataset

A small but useful observation. The validation split has noticeably lower variance than train and test. The reason is structural to SST-2 itself: the original training split contains a mixture of full sentences AND short sub-phrases from the parse tree, while the original validation split contains only full sentences. Our train and test were both carved from the original training split, while our val came from the original validation split — so val is more "uniform" in nature. This is a property of how SST-2 is constructed, not a problem with our pipeline, but it is worth knowing if anyone asks why the splits look slightly different in their statistics.

---

## 6. Visual evidence

The following plots are saved in `results/plots/` and form the core visual story:

- **`score_distribution_test.png`** — histogram of probe scores (`w · x + b`) for true-positive vs true-negative test sentences, on the same axes. Two clearly-separated humps, with the decision boundary in the middle. This is the visual proof of linear separability.
- **`confusion_test.png`** — heatmap of the test confusion matrix.
- **`roc_curve_test.png`** — ROC curve with AUC = 0.973.
- **`weight_magnitudes.png`** — histogram of `|w_i|` across the 2304 dimensions, showing that no single dimension dominates.

---

## 7. Bottom line

A single linear direction in Gemma 2 2B's layer-12 residual stream cleanly separates positive from negative SST-2 sentences with 92% test accuracy and AUC 0.97. This confirms that sentiment is represented linearly at this layer, but distributed across many dimensions. The next phase will compare this supervised direction against the unsupervised sentiment direction discovered by Gemma Scope's pretrained sparse autoencoder, to see whether the two methods agree on what sentiment looks like inside the model.
