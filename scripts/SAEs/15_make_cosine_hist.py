"""
Generate a histogram of cosine similarities between the linear probe direction
w and every SAE decoder row, with our top feature marked.

Output: results/plots/sae_all_cosines_hist.png
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from sae_lens import SAE

sns.set_theme(style="whitegrid", context="talk")

PLOTS = "results/plots"
os.makedirs(PLOTS, exist_ok=True)


def main():
    load_dotenv()
    token = os.getenv("HUGGING_FACE_TOKEN")
    if not token:
        sys.exit("HUGGING_FACE_TOKEN missing")
    login(token=token, add_to_git_credential=False)

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    sae = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res-canonical",
        sae_id="layer_12/width_16k/canonical",
        device=device,
    )
    sae.eval()
    W_dec = sae.W_dec.detach().cpu().to(torch.float32).numpy()  # (16384, 2304)

    w = np.load("results/probe_direction.npy").astype(np.float64)
    w_norm = np.linalg.norm(w)
    norms = np.linalg.norm(W_dec, axis=1)
    cos_all = (W_dec @ w) / (norms * w_norm)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(cos_all, bins=80, color="#4292c6", edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.8)
    cos14733 = float(cos_all[14733])
    ax.axvline(cos14733, color="#d62728", linewidth=2.5, linestyle="--",
               label=f"feature 14733 ({cos14733:+.3f})")
    ax.set_xlabel("cos(probe w, SAE decoder vector)")
    ax.set_ylabel("Number of SAE features")
    ax.set_title("Cosine similarity between probe direction and all 16,384 SAE features")
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = os.path.join(PLOTS, "sae_all_cosines_hist.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}  ({os.path.getsize(out)//1024} KB)")


if __name__ == "__main__":
    main()
