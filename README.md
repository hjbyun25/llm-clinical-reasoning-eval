<h1 align="center">
  Multi-Axial Analysis of Clinical Reasoning in Large Language Models:<br>
  Inter-Verifier Disagreement and Its Implications for Automated Evaluation
</h1>

## Structure

```
01_generation_and_semantic_similarity/   Sections 2.2, 2.3, 2.4.2, 4.1
02_medical_concept_grounding/            Section 2.4.1
03_semantic_uncertainty/                 Section 2.4.3
04_evidence_conclusion_coherence/        Section 2.4.4
05_statistical_analysis/                 Sections 2.4.5, 3.4, 3.5
```

Each folder contains a README named after the folder, with the method, the exact
command, and the part of the paper it corresponds to.

## Pipeline

| Step | Script |
|---|---|
| Generate five diagnosis-rationale outputs per case; compute reference similarity and inter-generation consistency | `01_generation_and_semantic_similarity/main.py` |
| Diagnosis-only sensitivity analysis | `01_generation_and_semantic_similarity/reference_similarity_full_vs_diagnosis.py` |
| Medical concept grounding against UMLS | `02_medical_concept_grounding/medical_concept_grounding.py` |
| Semantic entropy over clustered repeated generations | `03_semantic_uncertainty/semantic_uncertainty.py` |
| Coherence verification by three verifier LLMs | `04_evidence_conclusion_coherence/coherence_verification.ipynb` |
| Inter-verifier agreement and mean coherence scores | `05_statistical_analysis/inter_verifier_agreement.py` |
| Paired significance tests and effect sizes | `05_statistical_analysis/paired_significance_tests.py` |

## Data

The analysis uses MIMIC-IV, which is governed by the PhysioNet credentialed data
use agreement. 

## Credentials

No keys are included anywhere in this repository. Set the following in your
environment before running:

```bash
export HF_TOKEN=...          # gated HuggingFace models
export UMLS_API_KEY=...      # https://uts.nlm.nih.gov/uts/
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...
export OPENAI_API_KEY=...
```

## Embeddings

The two embedding models are not interchangeable; each axis uses a different one.

| Axis | Embedding |
|---|---|
| Semantic similarity (Eqs. 2-3) | `emilyalsentzer/Bio_ClinicalBERT` |
| Semantic uncertainty (Eq. 4) | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` |

## Reproducibility notes

- Generation uses `do_sample=True` with `temperature=0.7` and no fixed seed;
  rerunning will not reproduce the exact strings in our outputs.
- Grounding subsamples token pairs. Pass `--seed` for a deterministic subsample.
- Verifiers ran at `temperature=0`, but API models do not guarantee bitwise
  determinism; residual within-verifier variation cannot be fully excluded.
- Permutation p-values depend on the seed and the permutation count; they are
  stable to within Monte-Carlo error (~0.003 at 10,000 permutations). Wilcoxon
  p-values and the effect sizes are deterministic given the data.
