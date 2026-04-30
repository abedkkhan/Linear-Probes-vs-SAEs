# Tracker

A plain-English log of what we've done so far. One short paragraph per step.

---

## Step 0 — Environment setup

Created a clean Python virtual environment (`.venv`) using native arm64 Python 3.12 from Homebrew, since the default miniconda Python on this machine is Intel and can't install modern PyTorch on Apple Silicon. Installed the five core libraries we need to start: torch, transformers, datasets, scikit-learn, and numpy. Confirmed Apple's MPS GPU backend is available, which means model inference will run on the M4's GPU rather than the CPU. Added a `.gitignore` so the HuggingFace token in `.env` and the heavy `.venv` folder never get committed to git.

## Step 1 — Verify Gemma 2 2B loads

Wrote a small script that logs in to HuggingFace using the token from `.env`, downloads the Gemma 2 2B weights (about 10 GB, one-time), loads the model in half precision onto the MPS GPU, and runs one short test sentence through it. The script attaches a hook on layer 12 — the middle of the model's 26 layers — and confirms that the activation captured at that layer has the expected shape (one vector of 2,304 numbers per token). All checks passed, so we know authentication works, the GPU works, and the layer 12 hook is in the correct place.

## Step 2 — Prepare the dataset

Loaded the Stanford Sentiment Treebank (SST-2) from HuggingFace and discovered that its official `test` split has hidden labels (every label is `-1` because it's a leaderboard split). To work around this, we carved our own train (2000) and test (500) sets out of the original training data, and took our 500 validation examples from the official validation split. Sampling was done with numpy seeded at 42, so the exact same 3000 examples will be selected on any rerun. Saved everything as JSON files in `data/`, plus a manifest recording the seed and source indices for full reproducibility. Label distributions are roughly 55% positive / 45% negative — matching the source dataset's natural skew.

---

## Step 3a — Pilot extraction (smoke test)

Before running the full activation extraction on all 3000 sentences, we wrote a small "pilot" script that runs the exact same extraction logic on just the first 10 training sentences. The point was to surface every silent bug — wrong hook, wrong pooling axis, dtype mistakes, tokenizer surprises — in under a minute, rather than discovering them after a 10-minute extraction run on the full dataset. The script logs in to HuggingFace, loads Gemma 2 2B in half precision onto the MPS GPU, attaches a hook on layer 12 (the residual stream output of `Gemma2DecoderLayer`), feeds each of the 10 sentences through the model under `torch.no_grad()`, captures the layer-12 hidden states, mean-pools them across the sequence dimension to produce a single 2304-number vector per sentence, casts to float32, and prints diagnostics. It saves nothing to disk — it is a correctness check, not a production run.

The script also explicitly inspects the tokenization of the first sentence so we can see exactly what Gemma's tokenizer is doing, including any special tokens it adds. It then asserts four things that must be true for the pipeline to be valid: the pooled tensor has shape `(10, 2304)`, the dtype is `float32`, there are no NaN values, and there are no infinity values. If any of those fail, the script crashes loudly instead of silently producing garbage.

### What this lets us defend in the presentation

We can say, truthfully and precisely, that we are extracting the **output of the residual stream at layer 12** of a frozen Gemma 2 2B model — that is, the hidden state produced *after* the 13th decoder block has finished processing each token. We pool over the sequence dimension by simple arithmetic mean to get one fixed-size vector per sentence, because SST-2 phrases are short (4–11 tokens in the pilot) so mean pooling loses very little information, and because mean pooling is the standard, simplest, and most defensible aggregation in the linear probing literature. We compute everything in fp16 on GPU for memory efficiency, but cast to fp32 before saving so that downstream sklearn computations (StandardScaler, L2-regularized logistic regression) are not affected by half-precision rounding. Every random aspect of the experiment is upstream of this script — the sentence subset is fixed by seed 42 in step 1 — so this step is fully deterministic.

### Smoke test output

```
[1/6] Device: mps
[2/6] Model loaded. layers=26, hidden=2304
[3/6] Pilot: 10 sentences
      [0] 'as a director , eastwood is off his game'
      [0] 'skip this dreck ,'
      [1] 'fashioning an engrossing entertainment out'
      [1] 'bring tissues .'
      [1] 'cinematic bon bons'
      [0] 'irritates and'
      [0] 'derivative and hammily'
      [0] ', plodding picture'
      [0] 'an extremely unpleasant film .'
      [0] "as a dentist 's waiting room"
[5/6] First sentence tokenization:
      sentence: 'as a director , eastwood is off his game'
      n_tokens: 11
      ids:      [2, 508, 476, 8453, 1688, 9110, 4493, 603, 1401, 926, 2398]
      decoded:  ['<bos>', 'as', ' a', ' director', ' ,', ' east', 'wood', ' is', ' off', ' his', ' game']
[6/6] Diagnostics:
      raw shapes (per sentence): [(1, 11, 2304), (1, 6, 2304), (1, 9, 2304), (1, 4, 2304), (1, 5, 2304), (1, 4, 2304), (1, 6, 2304), (1, 5, 2304), (1, 6, 2304), (1, 8, 2304)]
      pooled tensor shape:        (10, 2304)  (expected (10, 2304))
      pooled dtype:               torch.float32
      labels:                     [0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
      any NaN?                    False
      any inf?                    False
      mean / std / min / max:     0.1743 / 10.1354 / -68.3125 / 604.0000
      first vector first 6 dims:  [-0.1959228515625, 0.19189453125, -1.26953125, 0.29931640625, -0.494384765625, -1.421875]
      |mean(pos) - mean(neg)|.mean(): 0.5565
All pilot checks passed.
```

### What this output tells us (and what to say if asked)

**The labels match the sentences.** Negatives like *"skip this dreck"*, *"derivative and hammily"*, *"an extremely unpleasant film"* are labeled 0; positives like *"fashioning an engrossing entertainment out"*, *"bring tissues"*, *"cinematic bon bons"* are labeled 1. This confirms our subsetting in step 1 preserved label alignment correctly.

**The tokenizer prepends a `<bos>` (beginning-of-sequence) token** with id 2. So an 11-token tokenization of a 9-word sentence is expected: 1 bos + 10 word/sub-word pieces. The word "eastwood" is split into two sub-word pieces (`east` + `wood`), which is normal for SentencePiece tokenizers. This means the `<bos>` activation is included in our mean pool. That is consistent with how almost all linear probing work on decoder LLMs is done, and is acceptable because (a) it appears in every sentence, so it cannot drive any *class-discriminative* signal, and (b) the same mean-pooling treatment is applied uniformly to every example for both the linear probe and the SAE pathway, so neither method gets an artificial advantage.

**The raw activation shapes are exactly `(1, seq_len, 2304)` for every sentence**, where `seq_len` varies between 4 and 11 tokens — proving the hook is firing in the right place and capturing the residual stream rather than something else (attention scores would be a different shape; logits would be `vocab_size = 256000`).

**After mean-pooling we get `(10, 2304)` in fp32**, exactly as designed. Ten sentences, one 2304-dimensional vector each, ready for the linear probe and the SAE.

**No NaN or inf values.** This is a real risk in fp16 inference on certain layers, so confirming it explicitly matters.

**The activation values are in a sensible range.** Mean is near zero (0.17), standard deviation is about 10, and the min/max span goes from -68 to +604. The fact that the maximum is unusually large (604) is normal for Gemma 2 — it has a small number of so-called "outlier features" in its residual stream that take very large values regardless of the input, a well-documented property of large LLMs. This is the precise reason the project plan specifies StandardScaler before the linear probe: it normalises every dimension to mean 0 and std 1, which prevents those outlier dimensions from dominating the L2-regularized fit.

**The per-class mean difference is 0.5565**, meaning when you take the average activation vector over the 2 positive examples, subtract the average over the 8 negative examples, and look at the average absolute difference per dimension, you get a non-zero value. With only 10 highly imbalanced samples this number is mostly noise, but the fact that it is non-zero confirms the pipeline is producing different vectors for different sentences — i.e. we are not accidentally returning identical embeddings. The real signal will be measured in step 3c on the full 3000.

### Likely presentation questions this prepares you for

- *"Why layer 12?"* — Middle layer of 26; semantic features (sentiment included) typically emerge in mid-layers of decoder LLMs (Alain & Bengio 2017; Park et al. 2024). We treat this as a starting point for the MVP and will sweep other layers in future work.
- *"Why mean pooling?"* — Standard in linear probing literature; SST-2 phrases are short so information loss is small; keeps the methodology simple and defensible.
- *"What about the BOS token contaminating the average?"* — It appears in every sample so it cannot create class-discriminative signal; both methods see the same pooled vectors so it does not bias the comparison.
- *"Is the hook on the right tensor?"* — Yes: it captures the first element of the tuple returned by `Gemma2DecoderLayer.forward`, which by HuggingFace convention is the residual stream output. Activation shapes match the model's `hidden_size`, which would not be the case for any other internal tensor.
- *"Why fp16 inference but fp32 on disk?"* — Memory savings during the forward pass, numerical safety for sklearn afterward.

---

## Step 3b — Full activation extraction

After the pilot confirmed everything was correct, we ran the same extraction logic on all 3000 sentences. The script loaded Gemma 2 2B once onto the MPS GPU, attached the hook on layer 12 once, and then walked through each split (train, val, test) one sentence at a time. For every sentence it tokenized, ran a forward pass under `torch.no_grad()`, captured the layer-12 residual stream, mean-pooled across the sequence dimension to get one 2304-number vector, and stored it in a pre-allocated numpy array. Before saving, every array was checked for NaN and inf values and asserted to have the expected shape, so a corrupt array can never silently overwrite a good one. End-to-end the run took about four minutes on the M4 — roughly 80 milliseconds per sentence — and produced six files in `data/`:

| File | What it contains | Shape | Size |
|---|---|---|---|
| `train_activations.npy` | 2000 sentence fingerprints (one row per sentence, 2304 numbers each) | `(2000, 2304)` fp32 | 18.4 MB |
| `train_labels.npy` | the matching positive/negative labels | `(2000,)` int64 | small |
| `val_activations.npy` | 500 fingerprints | `(500, 2304)` fp32 | 4.6 MB |
| `val_labels.npy` | matching labels | `(500,)` int64 | small |
| `test_activations.npy` | 500 fingerprints | `(500, 2304)` fp32 | 4.6 MB |
| `test_labels.npy` | matching labels | `(500,)` int64 | small |

These six files are everything the rest of the project needs. Gemma 2 2B is no longer required for any subsequent step — both the linear probe and the SAE analysis run directly on these saved numpy arrays, which means every downstream experiment now finishes in seconds rather than minutes. Activations and labels are stored side by side so they stay perfectly aligned: row `i` of any `*_activations.npy` is the fingerprint of the sentence whose label sits at position `i` of the matching `*_labels.npy`.

---

## Step 3c — Verification of saved activations

To make absolutely sure nothing went wrong during the long extraction run, we wrote a short verification script that loads the six saved files from disk and runs six checks on them. It confirms each array has the exact shape and dtype we expect, contains no NaN or inf values, and most importantly that the saved labels match the original JSON row by row — this is the check that would catch a silent label/activation misalignment, which is the single most dangerous bug in any ML pipeline because it does not crash anything but quietly destroys the experiment. It also reports the activation statistics for each split (mean, std, min, max), confirms the label distributions match what we saw when we built the JSON files, and asserts the per-class difference on the training set is non-zero, proving the pipeline is producing class-discriminative vectors. The script trains nothing and saves nothing — it is purely a sanity pass that runs in under a second.

---

## Where we are now

Phase 3 is complete. We have 3000 verified, correctly aligned, corruption-free 2304-dimensional fingerprints saved to disk, ready to be used by every downstream step. The next step is Phase 4: training the linear probe — a logistic regression classifier with L2 regularisation, fit on the standardised training fingerprints, evaluated on the held-out test set.
