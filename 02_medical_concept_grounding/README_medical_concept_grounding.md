# 2. Medical Concept Grounding

**Section 2.4.1.**

Tokens (alphabetic, length > 3, de-duplicated, capped at 60 per output) are
looked up in UMLS through the NLM UTS REST search API — a string lookup, not
MetaMap. The score is the proportion of sampled token pairs (up to 500 per
output) in which *both* tokens map to a concept, among pairs where at least one
does.

## Usage

```bash
export UMLS_API_KEY=...          # https://uts.nlm.nih.gov/uts/
python medical_concept_grounding.py \
    --input_dir ../generator_outputs --out_dir ./grounding --seed 42
```

Outputs a per-case grounding score and a per-model summary.
