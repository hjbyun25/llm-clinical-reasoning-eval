# 4. Evidence–Conclusion Coherence

**Section 2.4.4.**

Each scored generation is judged independently by three verifier LLMs under a
three-label scheme (ENTAILED / INSUFFICIENT / NOT_ENTAILED).

| Model | API identifier |
|---|---|
| Claude Sonnet 4.6 | `claude-sonnet-4-6` |
| Gemini 2.5 Pro | `gemini-2.5-pro` |
| GPT-5.4 mini | `gpt-5.4-mini` |

Accessed March 2026 at `temperature=0`. Gemini was called through the
OpenAI-compatible endpoint
(`generativelanguage.googleapis.com/v1beta/openai/chat/completions`).

The prompt requests a JSON object with `label`, `score` and `explanation`; only
`label` and `score` were retained for the reported analyses.

## Which generation was scored

A single fixed generation per case and generator was scored: the first of the
five repeated generations. These scripts expect as input a JSON file in which the five generations have already been reduced to a single generation for each case.

## Usage

Each verifier is run by its own script. Place the generator output JSONs under
`./generator_outputs/` and run:

```bash
export ANTHROPIC_API_KEY=...   # required by verifier_claude.py
export GOOGLE_API_KEY=...      # required by verifier_gemini.py
export OPENAI_API_KEY=...      # required by verifier_gpt.py

python verifier_claude.py
python verifier_gemini.py
python verifier_gpt.py
```

Output: `verification_{verifier}_{generator}.csv`, nine files in total.
