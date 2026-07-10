"""
Paired Significance Tests (Sections 2.4.5 and 3.4)

Paired Wilcoxon signed-rank and sign-flip permutation tests with
Holm-Bonferroni correction, plus matched-pairs rank-biserial correlation and
paired Cohen's d_z, for the between-generator and between-verifier contrasts.
See README_statistical_analysis.md for the method.

Usage:
    python paired_significance_tests.py \
        --data_dir <dir of verifier CSVs> --output paired_tests.csv \
        --n_perm 10000 --seed 42

Input: the nine verification_{verifier}_{generator}.csv files, each with
columns generator_model, case_id, sample_idx, label, score. The same 1,000
case_ids appear under every generator-verifier combination, so all comparisons
are matched by case_id.
"""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ----------------------------------------------------------------------
# Configuration: display names -> file-name fragments
#
# These fragments must match the names of the verifier output files on disk.
# Adjust them if your files are named differently.
# ----------------------------------------------------------------------
# Display name -> the fragment that appears in the verifier output filenames.
# These match the names written by coherence_verification.ipynb, i.e.
#   verification_{verifier_fragment}_{generator_fragment}.csv
GENERATORS = {
    "HuatuoGPT-o1-8B":            "HuatuoGPT-o1-8B",
    "Meta-Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "Meta-Llama-3.3-70B-Instruct": "Llama-3.3-70B",
}
VERIFIERS = {
    "Gemini 2.5 Pro":    "gemini-2.5-pro",
    "Claude Sonnet 4.6": "claude",
    "GPT-5.4 mini":      "gpt-5.4-mini",
}

# Contrast orderings as reported in the manuscript
GEN_PAIRS = [
    ("HuatuoGPT-o1-8B", "Meta-Llama-3.1-8B-Instruct"),
    ("HuatuoGPT-o1-8B", "Meta-Llama-3.3-70B-Instruct"),
    ("Meta-Llama-3.1-8B-Instruct", "Meta-Llama-3.3-70B-Instruct"),
]
VER_PAIRS = [
    ("Gemini 2.5 Pro", "Claude Sonnet 4.6"),
    ("Gemini 2.5 Pro", "GPT-5.4 mini"),
    ("Claude Sonnet 4.6", "GPT-5.4 mini"),
]


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_scores(data_dir):
    """Return dict[(verifier_name, generator_name)] -> pd.Series(score) indexed by case_id."""
    data = {}
    missing = []
    for v_name, v_frag in VERIFIERS.items():
        for g_name, g_frag in GENERATORS.items():
            path = Path(data_dir) / f"verification_{v_frag}_{g_frag}.csv"
            if not path.exists():
                missing.append(str(path))
                continue
            df = pd.read_csv(path)
            data[(v_name, g_name)] = df.set_index("case_id")["score"].astype(float)
    if missing:
        raise FileNotFoundError(
            "Missing verifier output file(s):\n  "
            + "\n  ".join(missing)
            + "\n\nAdjust the GENERATORS / VERIFIERS filename fragments at the top "
              "of this script to match your files."
        )
    return data


def aligned(a, b):
    """Align two case-indexed Series on their shared case_ids (paired design)."""
    idx = a.index.intersection(b.index)
    return a.loc[idx].values, b.loc[idx].values


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------
def wilcoxon_p(x, y):
    d = x - y
    if np.all(d == 0):
        return 1.0
    return stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided").pvalue


def permutation_p(x, y, n_perm=10000, seed=42):
    """Two-sided sign-flip permutation test on the mean paired difference."""
    rng = np.random.RandomState(seed)
    d = x - y
    d = d[d != 0]
    if len(d) == 0:
        return 1.0
    obs = abs(d.mean())
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([1, -1], size=len(d))
        if abs((d * signs).mean()) >= obs - 1e-12:
            count += 1
    return (count + 1) / (n_perm + 1)


def rank_biserial(x, y):
    """Matched-pairs rank-biserial correlation from the signed-rank statistic."""
    d = x - y
    nz = d[d != 0]
    if len(nz) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nz))
    r_pos = ranks[nz > 0].sum()
    r_neg = ranks[nz < 0].sum()
    total = r_pos + r_neg
    return (r_pos - r_neg) / total


def cohen_dz(x, y):
    d = x - y
    sd = d.std(ddof=1)
    return 0.0 if sd == 0 else d.mean() / sd


def holm_bonferroni(pvals):
    """Holm-Bonferroni step-down adjusted p-values (order preserved)."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(running, 1.0)
    return adj


# ----------------------------------------------------------------------
# Contrast families
# ----------------------------------------------------------------------
def run_family(data, pairs, fixed_names, fixed_kind, n_perm, seed):
    """
    fixed_kind = 'verifier'  -> compare generators within each verifier
    fixed_kind = 'generator' -> compare verifiers within each generator
    """
    rows = []
    for fixed in fixed_names:
        block = []
        praw = []
        for a, b in pairs:
            if fixed_kind == "verifier":
                xa, xb = data[(fixed, a)], data[(fixed, b)]
                stratum = fixed
                contrast = f"{a} vs {b}"
            else:
                xa, xb = data[(a, fixed)], data[(b, fixed)]
                stratum = fixed
                contrast = f"{a} vs {b}"
            x, y = aligned(xa, xb)
            block.append({
                "stratum": stratum,
                "contrast": contrast,
                "delta_MCS": x.mean() - y.mean(),
                "p_wilcoxon": wilcoxon_p(x, y),
                "p_perm": permutation_p(x, y, n_perm=n_perm, seed=seed),
                "r_rb": rank_biserial(x, y),
                "d_z": cohen_dz(x, y),
            })
            praw.append(block[-1]["p_perm"])
        padj = holm_bonferroni(praw)
        for r, pa in zip(block, padj):
            r["p_holm"] = pa
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=".")
    ap.add_argument("--output", default="paired_tests.csv")
    ap.add_argument("--n_perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = load_scores(args.data_dir)

    block1 = run_family(data, GEN_PAIRS, list(VERIFIERS.keys()), "verifier",
                        args.n_perm, args.seed)
    block2 = run_family(data, VER_PAIRS, list(GENERATORS.keys()), "generator",
                        args.n_perm, args.seed)

    df1 = pd.DataFrame(block1); df1.insert(0, "block", "between-generator (within verifier)")
    df2 = pd.DataFrame(block2); df2.insert(0, "block", "between-verifier (within generator)")
    out = pd.concat([df1, df2], ignore_index=True)

    for c in ["delta_MCS", "r_rb", "d_z"]:
        out[c] = out[c].round(3)
    for c in ["p_wilcoxon", "p_perm", "p_holm"]:
        out[c] = out[c].map(lambda v: f"{v:.3g}")

    out.to_csv(args.output, index=False)
    pd.set_option("display.width", 160, "display.max_columns", 20)
    print(out.to_string(index=False))
    print(f"\nSaved: {args.output}  (n_perm={args.n_perm}, seed={args.seed})")


if __name__ == "__main__":
    main()
