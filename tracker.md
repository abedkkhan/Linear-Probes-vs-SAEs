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

## Where we are now

We have a working language model, a confirmed correct hook target at layer 12, and a fixed set of 3000 labelled movie review snippets. The next step (Phase 3 in the project plan) is to feed each snippet through the model, listen at layer 12, mean-pool the activations into a single 2304-dimensional vector per sentence, and save those vectors as numpy arrays. Those vectors are the raw material that both the linear probe and the sparse autoencoder will analyse.
