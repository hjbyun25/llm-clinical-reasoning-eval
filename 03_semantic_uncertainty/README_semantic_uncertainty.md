# 3. Semantic Uncertainty

**Section 2.4.3.**

The five generations per case are embedded with BiomedBERT
(`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`), mean-pooled
and L2-normalized over the full generated output. Generations are grouped by
complete-linkage clustering on pairwise cosine similarity: a generation joins a
cluster only if its similarity to every member is at least tau. Semantic entropy
is the Shannon entropy (base 2) over the cluster-size distribution.

Computed for tau in {0.97, 0.98, 0.985, 0.99, 0.995}; tau = 0.97 is the primary
setting used throughout the main analyses.

## Usage

```bash
python semantic_uncertainty.py \
    --input ../generator_outputs/FreedomIntelligence_HuatuoGPT-o1-8B.json \
    --out_dir ./entropy
```

Outputs semantic entropy and the mean number of clusters at each tau.

> **Note.** This axis uses BiomedBERT, whereas semantic similarity (folder 01)
> uses Bio_ClinicalBERT. The `pairwise_mean` reported by this script is a
> BiomedBERT quantity and is *not* the inter-generation consistency of the
> similarity axis.
