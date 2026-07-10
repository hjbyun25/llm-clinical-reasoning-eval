# 1. Generation and Semantic Similarity

**Sections 2.2, 2.3, 2.4.2 and 4.1.**

| File | Role |
|---|---|
| `main.py` | Runs each generator five times per case; computes reference similarity and inter-generation consistency |
| `make_answer.py` | Wrapper dispatching to a local HuggingFace pipeline or the OpenAI API |
| `reference_similarity_full_vs_diagnosis.py` | Diagnosis-only sensitivity analysis (Section 4.1) |

## Generators

Accessed November 2025.

| Model | HuggingFace identifier |
|---|---|
| HuatuoGPT-o1-8B | `FreedomIntelligence/HuatuoGPT-o1-8B` |
| Meta-Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` |
| Meta-Llama-3.3-70B-Instruct | `meta-llama/Llama-3.3-70B-Instruct` |

Decoding, set once when the pipeline is constructed in `main.py`:
`temperature=0.7`, `top_p=0.9`, `top_k=50`, `repetition_penalty=1.2`,
`max_new_tokens=512`, `do_sample=True`, `return_full_text=False`.

## Semantic similarity (Section 2.4.2)

`main.py` embeds text with Bio_ClinicalBERT (`emilyalsentzer/Bio_ClinicalBERT`,
`normalize_embeddings=True`) and stores two quantities per case in the output
JSON.

| Field | Definition |
|---|---|
| `semantic_similarity` | Cosine similarity between each generation and the reference diagnosis, averaged over the five generations (Eq. 2) |
| `semantic_consistency` | Mean pairwise cosine similarity over the ten unordered pairs among a case's five generations (Eq. 3) |

A third field, `semantic_uncertainty`, is stored as `1 - semantic_consistency`.
It is a by-product of this script and is distinct from the semantic entropy of
folder `03_semantic_uncertainty`, which the paper reports.

## Diagnosis-only sensitivity analysis (Section 4.1)

`reference_similarity_full_vs_diagnosis.py` recomputes reference similarity on
the extracted diagnoses alone, using the same embedding and procedure as
`main.py`. Added in response to Reviewer 1, Comment 1.7.

## Usage

```bash
export HF_TOKEN=...
python main.py --model meta-llama/Llama-3.3-70B-Instruct --temp 0.7 \
    --data ./test_data.csv --out_dir ./result

python reference_similarity_full_vs_diagnosis.py
```

Input: `test_data.csv` — indexed by case ID, with columns
`patient_info` (gender, race, age), `HPI`, and `diagnosis`. Only `patient_info`
and `HPI` are shown to the model; `diagnosis` is withheld from the input and
used solely as the reference for the similarity scores.
Output: `result/{model}_result.json`, with five `predictions` per case.
