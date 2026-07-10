"""
Inter-Verifier Agreement (Sections 3.4 and 3.5)

Computes Fleiss' kappa, pairwise Cohen's kappa, the disagreement rate, and the
Mean Coherence Score with bootstrap 95% CIs.
See README_statistical_analysis.md for the method.

Usage:
    python inter_verifier_agreement.py \
        --verifier_data <per-generator CSV or JSON> \
        --metric_data <generator output JSON> --output_dir ./stats

Input: one file per generator, with one column per verifier holding
ENTAILED / INSUFFICIENT / NOT_ENTAILED. Verifier column names are read as-is.
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


LABEL_MAP = {"ENTAILED": 0, "INSUFFICIENT": 1, "NOT_ENTAILED": 2}
LABEL_NAMES = ["ENTAILED", "INSUFFICIENT", "NOT_ENTAILED"]
SCORE_MAP = {"ENTAILED": 1.0, "INSUFFICIENT": 0.5, "NOT_ENTAILED": 0.0}


# ============================================================
# Data Loading (auto-detect JSON or CSV)
# ============================================================

def load_verifier_data(path):
    path = Path(path)
    if path.suffix == ".json":
        with open(path) as f:
            raw = json.load(f)
        rows = []
        for case_id, verifiers in raw.items():
            row = {"case_id": case_id}
            for v_name, label in verifiers.items():
                row[v_name] = label.strip().upper()
            rows.append(row)
        df = pd.DataFrame(rows).set_index("case_id")
    elif path.suffix == ".csv":
        df = pd.read_csv(path, index_col=0)
        for c in df.columns:
            df[c] = df[c].map(lambda x: str(x).strip().upper() if isinstance(x, str) else x)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use .json or .csv")

    verifier_cols = [c for c in df.columns if c != "case_id"]
    print(f"Loaded verifier data: {len(df)} cases, {len(verifier_cols)} verifiers: {verifier_cols}")
    return df, verifier_cols


def load_metric_data(path):
    path = Path(path)
    if path.suffix == ".json":
        with open(path) as f:
            raw = json.load(f)
        if isinstance(raw, dict) and all(isinstance(v, dict) for v in raw.values()):
            df = pd.DataFrame.from_dict(raw, orient="index")
        else:
            df = pd.DataFrame(raw)
    elif path.suffix == ".csv":
        df = pd.read_csv(path, index_col=0)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    print(f"Loaded metric data: {len(df)} cases, columns: {list(df.columns)}")
    return df


# ============================================================
# 1. Fleiss' Kappa
# ============================================================

def compute_fleiss_kappa(df, verifier_cols):
    n_categories = len(LABEL_NAMES)
    rating_matrix = np.zeros((len(df), n_categories), dtype=int)

    for i, (_, row) in enumerate(df.iterrows()):
        for col in verifier_cols:
            label = row[col]
            if label in LABEL_MAP:
                rating_matrix[i, LABEL_MAP[label]] += 1

    N = rating_matrix.shape[0]
    n = rating_matrix.sum(axis=1)[0]
    p_j = rating_matrix.sum(axis=0) / (N * n)
    P_i = (np.sum(rating_matrix ** 2, axis=1) - n) / (n * (n - 1))
    P_bar = np.mean(P_i)
    P_e = np.sum(p_j ** 2)

    if P_e == 1.0:
        kappa = 1.0
    else:
        kappa = (P_bar - P_e) / (1 - P_e)

    return kappa, P_bar, P_e


def pairwise_cohen_kappa(df, col_a, col_b):
    labels_a = df[col_a].map(LABEL_MAP).values
    labels_b = df[col_b].map(LABEL_MAP).values
    valid = ~(np.isnan(labels_a) | np.isnan(labels_b))
    labels_a, labels_b = labels_a[valid].astype(int), labels_b[valid].astype(int)

    n = len(labels_a)
    n_cat = len(LABEL_NAMES)
    confusion = np.zeros((n_cat, n_cat), dtype=int)
    for a, b in zip(labels_a, labels_b):
        confusion[a, b] += 1

    p_o = np.trace(confusion) / n
    row_sums = confusion.sum(axis=1) / n
    col_sums = confusion.sum(axis=0) / n
    p_e = np.sum(row_sums * col_sums)

    if p_e == 1.0:
        return 1.0, confusion
    return (p_o - p_e) / (1 - p_e), confusion


def disagreement_rate(df, verifier_cols):
    n_disagree = 0
    for _, row in df.iterrows():
        labels = set(row[col] for col in verifier_cols)
        if len(labels) > 1:
            n_disagree += 1
    return n_disagree / len(df)


def verifier_agreement_analysis(df, verifier_cols):
    results = {}

    kappa, P_bar, P_e = compute_fleiss_kappa(df, verifier_cols)
    results["fleiss_kappa"] = kappa
    results["P_bar"] = P_bar
    results["P_e"] = P_e

    results["disagreement_rate"] = disagreement_rate(df, verifier_cols)

    from itertools import combinations
    pairwise = {}
    for a, b in combinations(verifier_cols, 2):
        k, conf = pairwise_cohen_kappa(df, a, b)
        pairwise[f"{a} vs {b}"] = {
            "cohen_kappa": k,
            "confusion_matrix": conf.tolist()
        }
    results["pairwise_cohen_kappa"] = pairwise

    return results


# ============================================================
# 2. Bootstrap CI
# ============================================================

def bootstrap_ci(values, n_boot=10000, ci=95, seed=42):
    rng = np.random.RandomState(seed)
    values = np.array(values, dtype=float)
    values = values[~np.isnan(values)]
    boot_means = np.array([np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)])
    lo = np.percentile(boot_means, (100 - ci) / 2)
    hi = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return float(np.mean(values)), float(lo), float(hi)


def compute_metric_ci(df):
    results = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        mean, lo, hi = bootstrap_ci(df[col].dropna().values)
        results[col] = {"mean": mean, "ci_low": lo, "ci_high": hi, "formatted": f"{mean:.4f} [{lo:.4f}, {hi:.4f}]"}
    return results


def compute_mcs_ci(df, verifier_cols):
    results = {}
    for col in verifier_cols:
        scores = df[col].map(SCORE_MAP).dropna().values
        mean, lo, hi = bootstrap_ci(scores)
        results[col] = {"mcs": mean, "ci_low": lo, "ci_high": hi, "formatted": f"{mean:.4f} [{lo:.4f}, {hi:.4f}]"}
    return results


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Statistical analysis: Fleiss' kappa + Bootstrap CI")
    ap.add_argument("--verifier_data", type=str, default=None, help="Path to verifier results (JSON or CSV)")
    ap.add_argument("--metric_data", type=str, default=None, help="Path to evaluation metrics (JSON or CSV)")
    ap.add_argument("--output_dir", type=str, default="./stats")
    ap.add_argument("--n_boot", type=int, default=10000)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    if args.verifier_data:
        print("\n" + "=" * 60)
        print("1. INTER-VERIFIER AGREEMENT ANALYSIS")
        print("=" * 60)

        df_v, v_cols = load_verifier_data(args.verifier_data)
        agreement = verifier_agreement_analysis(df_v, v_cols)

        print(f"\n  Fleiss' kappa:      {agreement['fleiss_kappa']:.4f}")
        print(f"  Disagreement rate:  {agreement['disagreement_rate']:.4f} ({agreement['disagreement_rate']*100:.1f}%)")
        print(f"\n  Pairwise Cohen's kappa:")
        for pair, info in agreement["pairwise_cohen_kappa"].items():
            print(f"    {pair}: {info['cohen_kappa']:.4f}")

        mcs_ci = compute_mcs_ci(df_v, v_cols)
        print(f"\n  MCS with 95% CI:")
        for col, info in mcs_ci.items():
            print(f"    {col}: {info['formatted']}")

        agreement["mcs_bootstrap_ci"] = {k: {kk: vv for kk, vv in v.items() if kk != "formatted"} for k, v in mcs_ci.items()}
        all_results["verifier_agreement"] = agreement

        agreement_path = output_dir / "verifier_agreement.json"
        with open(agreement_path, "w") as f:
            json.dump(agreement, f, indent=2)
        print(f"\n  Saved: {agreement_path}")

    if args.metric_data:
        print("\n" + "=" * 60)
        print("2. BOOTSTRAP CI FOR EVALUATION METRICS")
        print("=" * 60)

        df_m = load_metric_data(args.metric_data)
        metric_ci = compute_metric_ci(df_m)

        print(f"\n  Metrics with 95% CI:")
        for col, info in metric_ci.items():
            print(f"    {col}: {info['formatted']}")

        all_results["metric_bootstrap_ci"] = {k: {kk: vv for kk, vv in v.items() if kk != "formatted"} for k, v in metric_ci.items()}

        ci_path = output_dir / "bootstrap_ci.json"
        with open(ci_path, "w") as f:
            json.dump(all_results["metric_bootstrap_ci"], f, indent=2)
        print(f"\n  Saved: {ci_path}")

    if all_results:
        summary_path = output_dir / "full_analysis.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n{'=' * 60}")
        print(f"Full analysis saved: {summary_path}")

    if not args.verifier_data and not args.metric_data:
        print("No data provided. Use --verifier_data and/or --metric_data.")
        print("\nExpected formats:")
        print("\n  Verifier JSON:")
        print('  {"case_001": {"gemini 2.5 pro": "ENTAILED", "claude sonnet 4.6": "INSUFFICIENT", "gpt 5.4 mini": "ENTAILED"}, ...}')
        print("\n  Verifier CSV:")
        print("  case_id,gemini 2.5 pro,claude sonnet 4.6,gpt 5.4 mini")
        print("  case_001,ENTAILED,INSUFFICIENT,ENTAILED")
        print("\n  Metric JSON (from main.py output):")
        print('  {"0": {"semantic_similarity": 0.82, "semantic_consistency": 0.95, ...}, ...}')


if __name__ == "__main__":
    main()
