"""
Semantic Uncertainty (Section 2.4.3)

Computes the semantic entropy of each case's five repeated generations.
See README_semantic_uncertainty.md for the method.

Usage:
    python semantic_uncertainty.py \
        --input <generator output JSON> --out_dir ./entropy

Outputs (per input file):
    {base}_sentence_uncertainty.json   per-case statistics
    {base}_sentence_case.csv           one row per (case, tau)
    {base}_sentence_summary.csv        mean entropy / n_clusters / pairwise_mean by tau
"""

import argparse
import json
import math
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
TAUS = [0.97, 0.98, 0.985, 0.99, 0.995]
MAX_LENGTH = 256
BATCH_SIZE = 16


def mean_pool(last_hidden, mask):
    mask = mask.unsqueeze(-1).float()
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


@torch.no_grad()
def embed(texts, tokenizer, model, device):
    embs = []
    for i in range(0, len(texts), BATCH_SIZE):
        enc = tokenizer(
            texts[i : i + BATCH_SIZE],
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)
        out = model(**enc)
        e = mean_pool(out.last_hidden_state, enc["attention_mask"])
        e = torch.nn.functional.normalize(e, dim=1)
        embs.append(e.cpu().numpy())
    return np.vstack(embs)


def cosine_sim(E):
    return E @ E.T


def complete_linkage(sim, tau):
    """A generation joins a cluster only if its similarity to every member >= tau."""
    clusters = []
    for i in range(sim.shape[0]):
        placed = False
        for c in clusters:
            if all(sim[i, j] >= tau for j in c):
                c.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])
    return clusters


def entropy(clusters, K):
    """Shannon entropy (base 2) over the cluster-size distribution."""
    ps = [len(c) / K for c in clusters]
    return -sum(p * math.log2(p) for p in ps if p > 0)


def tri_stats(sim):
    iu = np.triu_indices(sim.shape[0], k=1)
    vals = sim[iu]
    return float(vals.min()), float(vals.mean()), float(vals.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="generator output JSON")
    ap.add_argument("--out_dir", default="./entropy")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(args.input))[0]
    out_json = os.path.join(args.out_dir, f"{base}_sentence_uncertainty.json")
    out_case = os.path.join(args.out_dir, f"{base}_sentence_case.csv")
    out_summary = os.path.join(args.out_dir, f"{base}_sentence_summary.csv")

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    rows = []
    for cid, obj in tqdm(data.items(), desc="Sentence-level"):
        preds = obj["predictions"]
        E = embed(preds, tokenizer, model, device)
        S = cosine_sim(E)
        smin, smean, smax = tri_stats(S)

        obj["sentence_uncertainty"] = {
            "pairwise_min": smin,
            "pairwise_mean": smean,
            "pairwise_max": smax,
            "per_tau": {},
        }

        for tau in TAUS:
            clusters = complete_linkage(S, tau)
            H = entropy(clusters, len(preds))
            obj["sentence_uncertainty"]["per_tau"][str(tau)] = {
                "n_clusters": len(clusters),
                "cluster_sizes": [len(c) for c in clusters],
                "semantic_entropy": H,
            }
            rows.append(
                {
                    "case_id": cid,
                    "tau": tau,
                    "n_clusters": len(clusters),
                    "semantic_entropy": H,
                    "pairwise_min": smin,
                    "pairwise_mean": smean,
                    "pairwise_max": smax,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_case, index=False)
    df.groupby("tau")[["semantic_entropy", "n_clusters", "pairwise_mean"]].mean().reset_index().to_csv(
        out_summary, index=False
    )
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("\nSummary by tau:")
    print(df.groupby("tau")[["semantic_entropy", "n_clusters"]].mean().round(4))


if __name__ == "__main__":
    main()
