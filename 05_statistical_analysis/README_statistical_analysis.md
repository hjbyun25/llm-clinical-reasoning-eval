# 5. Statistical Analysis

**Sections 2.4.5, 3.4 and 3.5.**

| File | Produces |
|---|---|
| `inter_verifier_agreement.py` | Mean coherence scores with bootstrap CIs; Fleiss' kappa, pairwise Cohen's kappa, disagreement rate |
| `paired_significance_tests.py` | Paired Wilcoxon and permutation tests, Holm–Bonferroni correction, effect sizes |

## Inter-verifier agreement

```bash
python inter_verifier_agreement.py \
    --verifier_data <per-generator CSV> --output_dir ./stats
```

Expects one CSV or JSON per generator, with one column per verifier holding
ENTAILED / INSUFFICIENT / NOT_ENTAILED. Verifier column names are read as-is.

## Paired significance tests

```bash
python paired_significance_tests.py \
    --data_dir <dir> --output paired_tests.csv --n_perm 10000 --seed 42
```

Expects the nine `verification_{verifier}_{generator}.csv` files written by the
coherence notebook. Adjust the `GENERATORS` / `VERIFIERS` filename fragments at
the top of the script if your files are named differently.

Permutation p-values depend on the seed and the permutation count; they are
stable to within Monte-Carlo error (~0.003 at 10,000 permutations). Wilcoxon
p-values and the effect sizes are deterministic given the data.
