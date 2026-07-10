"""
Medical Concept Grounding (Section 2.4.1)

Looks tokens up in UMLS through the NLM UTS REST search API and scores the
proportion of sampled token pairs in which both tokens map to a concept.
See README_medical_concept_grounding.md for the method.

Credentials:
    export UMLS_API_KEY=...      # https://uts.nlm.nih.gov/uts/

Usage:
    python medical_concept_grounding.py \
        --input_dir <dir of generator output JSONs> --out_dir ./grounding --seed 42

Outputs:
    {model}_grounding_scores.csv    one row per (case, generation)
    {model}_grounding_summary.csv   per-model summary
    ALL_MODELS_grounding_scores.csv / _summary.csv
"""

import argparse
import json
import os
import random
import re
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

UMLS_API_KEY = os.environ.get("UMLS_API_KEY", "").strip()
if not UMLS_API_KEY:
    raise RuntimeError("Set the UMLS_API_KEY environment variable before running.")

UMLS_BASE = "https://uts-ws.nlm.nih.gov/rest"

# Cache term -> (matched, semantic_types); the same terms recur constantly.
TERM_CACHE = {}

SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json"})


def umls_search_term(term: str):
    """Look a single token up in UMLS. Returns (matched, semantic_types)."""
    term = term.lower().strip()
    if not term:
        return False, []
    if term in TERM_CACHE:
        return TERM_CACHE[term]

    try:
        resp = SESSION.get(
            f"{UMLS_BASE}/search/current",
            params={"apiKey": UMLS_API_KEY, "string": term, "pageSize": 3},
            timeout=3,
        )
        if resp.status_code != 200:
            TERM_CACHE[term] = (False, [])
            return False, []

        results = resp.json().get("result", {}).get("results", [])
        if not results:
            TERM_CACHE[term] = (False, [])
            return False, []

        cui = results[0].get("ui")
        if cui in (None, "NONE"):
            TERM_CACHE[term] = (False, [])
            return False, []

        # Semantic types are retrieved for inspection but do not enter the score.
        st_resp = SESSION.get(
            f"{UMLS_BASE}/content/current/CUI/{cui}",
            params={"apiKey": UMLS_API_KEY},
            timeout=3,
        )
        if st_resp.status_code != 200:
            TERM_CACHE[term] = (True, [])
            return True, []

        semtypes = [
            x.get("name")
            for x in st_resp.json().get("result", {}).get("semanticTypes", [])
            if x.get("name")
        ]
        TERM_CACHE[term] = (True, semtypes)
        return True, semtypes

    except Exception:
        TERM_CACHE[term] = (False, [])
        return False, []


def tokenize_for_medical(text: str, max_tokens: int = 60):
    """Alphabetic tokens longer than three characters, de-duplicated, capped."""
    tokens = re.findall(r"[A-Za-z]+", (text or "").lower())
    tokens = [t for t in tokens if len(t) > 3]
    tokens = list(dict.fromkeys(tokens))  # de-duplicate, preserve order
    return tokens[:max_tokens]


def grounding_score(prediction_text: str, max_tokens: int = 60, max_pairs: int = 500):
    """Proportion of candidate pairs in which both tokens map to UMLS concepts."""
    tokens = tokenize_for_medical(prediction_text, max_tokens)
    if len(tokens) < 2:
        return 0.0

    umls_hit = {t: umls_search_term(t)[0] for t in tokens}

    # Denominator: pairs where at least one token is a UMLS concept.
    candidate_pairs = [
        (tokens[i], tokens[j])
        for i in range(len(tokens))
        for j in range(i + 1, len(tokens))
        if umls_hit[tokens[i]] or umls_hit[tokens[j]]
    ]
    if not candidate_pairs:
        return 0.0

    # Monte-Carlo subsample for tractability across varying text lengths.
    if len(candidate_pairs) > max_pairs:
        candidate_pairs = random.sample(candidate_pairs, max_pairs)

    match = sum(1 for t1, t2 in candidate_pairs if umls_hit[t1] and umls_hit[t2])
    return match / len(candidate_pairs)


def evaluate_file(json_path, model_name, out_dir):
    with open(json_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    rows = []
    for case_id, content in tqdm(data.items(), desc=f"Evaluating {model_name}"):
        for pred_id, pred in enumerate(content.get("predictions", [])):
            rows.append(
                {
                    "model": model_name,
                    "case_id": case_id,
                    "pred_id": pred_id,
                    "grounding_score": grounding_score(pred),
                }
            )

    df = pd.DataFrame(rows)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f"{model_name}_grounding_scores.csv"), index=False)
    df.describe().to_csv(os.path.join(out_dir, f"{model_name}_grounding_summary.csv"))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="./generator_outputs")
    ap.add_argument("--out_dir", default="./grounding")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    files = {
        "HuatuoGPT-o1-8B": "FreedomIntelligence_HuatuoGPT-o1-8B.json",
        "Llama-3.1-8B": "meta-llama_Llama-3.1-8B-Instruct.json",
        "Llama-3.3-70B": "meta-llama_Llama-3.3-70B-Instruct.json",
    }

    all_df = []
    for model, fname in files.items():
        all_df.append(evaluate_file(os.path.join(args.input_dir, fname), model, args.out_dir))

    total = pd.concat(all_df, ignore_index=True)
    total.to_csv(os.path.join(args.out_dir, "ALL_MODELS_grounding_scores.csv"), index=False)
    total.groupby("model")[["grounding_score"]].describe().to_csv(
        os.path.join(args.out_dir, "ALL_MODELS_grounding_summary.csv")
    )
    print("\nPer-model mean grounding score:")
    print(total.groupby("model")["grounding_score"].mean().round(4))


if __name__ == "__main__":
    main()
