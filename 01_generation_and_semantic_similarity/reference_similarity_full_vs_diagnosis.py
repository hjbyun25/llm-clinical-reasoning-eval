# ============================================================
# Reference Similarity: FULL vs DIAG  (Bio_ClinicalBERT)
# ============================================================
# Purpose: using the same embedding (Bio_ClinicalBERT) and the same procedure
#   as main.py, compare reference similarity computed on the FULL generated
#   output against the extracted DIAGNOSES only.
#   Reported in Section 4.1 (response to Reviewer 1, Comment 1.7).
#
# Identical to main.py: SentenceTransformer("emilyalsentzer/Bio_ClinicalBERT"),
#   normalize_embeddings=True, util.cos_sim, averaged over the five generations.
# This is post-processing of existing generator outputs; generation is not rerun.
# ============================================================

import os, json, re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# -------------------------
# Settings (identical to main.py)
# -------------------------
EMBED_MODEL = "emilyalsentzer/Bio_ClinicalBERT"

FILES = {
    "HuatuoGPT-o1-8B": "FreedomIntelligence_HuatuoGPT-o1-8B.json",
    "Llama-3.1-8B":    "meta-llama_Llama-3.1-8B-Instruct.json",
    "Llama-3.3-70B":   "meta-llama_Llama-3.3-70B-Instruct.json",
}
OUT_DIR = "."

# -------------------------
# Extract primary + secondary diagnoses
# -------------------------
PRIMARY_PAT = re.compile(
    r'(?:^|\n)\s*\**\s*primary\s+diagnos[ie]s?\**\s*:?\s*(.*?)'
    r'(?=\n\s*\**\s*(?:secondary|rationale|note)\b|\Z)',
    re.IGNORECASE | re.DOTALL
)
SECONDARY_PAT = re.compile(
    r'(?:^|\n)\s*\**\s*secondary\s+diagnos[ie]s?\**\s*:?\s*(.*?)'
    r'(?=\n\s*\**\s*(?:rationale|note|primary)\b|\Z)',
    re.IGNORECASE | re.DOTALL
)

def _clean(block):
    if not block: return ""
    lines = []
    for ln in block.splitlines():
        s = ln.strip()
        s = re.sub(r'^[\d\.\)\-\*\u2022\s]+', '', s)
        s = re.sub(r'\*+', '', s).strip(' .;:')
        if s: lines.append(s)
    return ", ".join(lines).strip()

def extract_diag(text):
    text = str(text)
    pm = PRIMARY_PAT.search(text)
    sm = SECONDARY_PAT.search(text)
    primary = _clean(pm.group(1)) if pm else ""
    secondary = _clean(sm.group(1)) if sm else ""
    parts = [p for p in [primary, secondary] if p]
    return (" ; ".join(parts)) if parts else None

# -------------------------
# Same similarity computation as main.py
# -------------------------
def sim(embedder, a, b):
    ea = embedder.encode(a, normalize_embeddings=True)
    eb = embedder.encode(b, normalize_embeddings=True)
    return float(util.cos_sim(ea, eb).item())

# -------------------------
# Main
# -------------------------
def run():
    print(f"Loading {EMBED_MODEL} (SentenceTransformer) ...")
    embedder = SentenceTransformer(EMBED_MODEL)

    rows = []
    for model_name, path in FILES.items():
        if not os.path.exists(path):
            print(f"[SKIP] {path}"); continue
        data = json.load(open(path, "r", encoding="utf-8-sig"))

        full_case_means, diag_case_means = [], []
        n_diag_cases = 0

        for cid, obj in data.items():
            preds = obj.get("predictions", [])
            true_diag = str(obj.get("true_diagnosis", "")).strip()
            if not preds or not true_diag:
                continue

            # FULL: whole generated output vs reference (as in main.py)
            full_sims = [sim(embedder, str(p), true_diag) for p in preds]
            full_case_means.append(np.mean(full_sims))

            # DIAG: extracted diagnoses only vs reference (extractable cases)
            diags = [extract_diag(p) for p in preds]
            diags = [d for d in diags if d]
            if diags:
                diag_sims = [sim(embedder, d, true_diag) for d in diags]
                diag_case_means.append(np.mean(diag_sims))
                n_diag_cases += 1

        rows.append({
            "model": model_name,
            "ref_sim_FULL": round(np.mean(full_case_means), 4) if full_case_means else None,
            "ref_sim_DIAG": round(np.mean(diag_case_means), 4) if diag_case_means else None,
            "n_cases_FULL": len(full_case_means),
            "n_cases_DIAG_extractable": n_diag_cases,
        })

    df = pd.DataFrame(rows)
    print("\n" + "="*70)
    print("Reference Similarity (Bio_ClinicalBERT): FULL vs DIAG")
    print("="*70)
    print(df.to_string(index=False))
    print("\n[Expected FULL values] Huatuo=0.775, Llama-3.1=0.800, Llama-3.3=0.804")
    print("If ref_sim_FULL matches, the FULL computation is reproduced and the DIAG comparison is valid.")
    df.to_csv(os.path.join(OUT_DIR, "ref_sim_full_vs_diag_BioClinicalBERT.csv"), index=False)
    print("\nSaved: ref_sim_full_vs_diag_BioClinicalBERT.csv")

if __name__ == "__main__":
    run()
