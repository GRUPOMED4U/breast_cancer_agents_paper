# LLM-as-a-judge: protocol and validation (issue #11)

Working notes for the Methods, Supplement and response to reviewers
(R1-M2, R4-M2, R3-M9). Everything below was checked against the code and the
stored grades in `evals/`, unless marked **to confirm**. No patient data in this
file.

## A. Judging protocol (as run for the reported results)

| Item | Value | Source |
|---|---|---|
| Judge model | **Gemini 2.5 Flash** (`gemini-2.5-flash`, Google GenAI API) | `eval_model_id` is `gemini-2.5-flash` in all 27 files in `evals/` |
| Model version | Stable alias `gemini-2.5-flash`; the served version string was not recorded | **to confirm** if available from API logs; recorded from now on (see below) |
| Grading dates | 8 June – 1 July 2026 (file timestamps of `evals/*.json`) | file system; **to confirm** |
| Judge is also an evaluated model | Yes: Gemini 2.5 Flash is one of the five generator models | `dvc.yaml` |
| Prompt | `GRADER_TEMPLATE` in `evaluator.py` (verbatim below), adapted from HealthBench, with one added rule for missing information | `evaluator.py` |
| Output format | JSON `{"explanation": str, "criteria_met": bool}` enforced through the API's structured output (`response_mime_type=application/json`, `response_json_schema`) | `evaluator.py` |
| Sampling | No temperature, top-p or seed set: provider defaults apply | `evaluator.py` |
| Reasoning | No thinking budget set, so Gemini 2.5 Flash's default dynamic thinking was on. Mean ≈ 520 reasoning tokens per call (16.2M reasoning tokens over 30,969 calls) | `evaluation_token_usage` in `evals/` |
| Unit of grading | One API call per (response, criterion). 72 cases × 27 configurations × all criteria = **30,969 calls**, 78.0M input tokens | `evals/` |
| Repeats | One verdict per pair; no repeated grading | `evaluator.py` |
| Parse failures | Would be recorded silently as "not met". **None occurred**: all 30,969 verdicts have a non-empty explanation | `validation/judge_failure_modes.py` |

**What the judge sees, per call**
- the case summary (the same prompt the evaluated system received);
- the full system response (for the multi-agent pipelines, the final answer as stored, i.e. the Python repr of `{'text': ...}`);
- **one** rubric criterion, passed as the repr of the `Rubric` object. The judge therefore sees the criterion text, **its point value (and sign)** and **its category tag**.

**What the judge does not see**: the reference (tumor-board) decision, the other rubric criteria, the configuration (model or pipeline) that produced the response, and other responses to the same case. Each criterion is graded independently.

The Methods currently say only "a grader model". Proposed text:

> Responses were graded by Gemini 2.5 Flash (Google; accessed June–July 2026) acting as judge, using the HealthBench grading prompt with one added instruction on missing information (Supplementary Note X, verbatim). Each rubric criterion was graded in a separate call in which the judge saw the case summary, the full response and that single criterion with its point value and category, and returned a structured JSON verdict (met / not met) with an explanation. The judge did not see the reference decision, the other criteria, or the identity of the system that produced the response. Provider default sampling settings were used, with the model's default dynamic reasoning enabled; each criterion was graded once (30,969 judgments in total). Because Gemini 2.5 Flash was also one of the evaluated models, we tested for self-preference by re-grading with judges from other providers (Section Y).

### Changes made for reproducibility
- `evaluator.build_grader_prompt()` builds the exact prompt; re-grades reuse it. Prompts rebuilt from `evals/` were checked to be byte-identical to those built from `results/` + `data/eval_dataset.jsonl`.
- `run_eval.py` accepts `--grader-temperature`, `--grader-seed` and `--grader-thinking-budget` (default: unset, as in the paper) and saves them in the eval file as `grader_settings`, together with the served model version(s) (`grader_model_versions`).

### Verbatim prompt (for the Supplement)
See `GRADER_TEMPLATE` in [`evaluator.py`](../evaluator.py). Placeholders: `{query}` = case summary prompt, `{agent_response}` = response, `{rubric}` = one criterion.

## F. Known judge failure modes (computed, no API cost)

Script: `validation/judge_failure_modes.py` → `validation/results/judge_failure_modes.json`.

**Response length.** Within each case, the 27 responses were split into length quintiles.

| Within-case length quintile | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---|---|---|---|
| Median words | 484 | 610 | 767 | 1,142 | 1,706 |
| Met-rate, positive criteria | 0.43 | 0.48 | 0.50 | 0.56 | 0.60 |
| Met-rate, negative criteria | 0.08 | 0.09 | 0.09 | 0.08 | 0.09 |

- Positive criteria: met-rate rises with length (mean within-criterion Spearman ρ = 0.22, 95% CI 0.20–0.25; within the same pipeline ρ = 0.30, 0.27–0.34). Within a case, longer responses get higher case scores (ρ = 0.35, 0.30–0.41).
- Negative criteria: no association (ρ = 0.01, −0.03 to 0.04). Longer responses are not penalized more often.
- Interpretation: longer responses cover more of the recommended items, so a rise on positive criteria is expected even from a perfect judge. These data cannot separate real coverage from a verbosity bias. The clinician labels (option B, #14) can: test whether judge–clinician disagreement (judge "met", clinician "not met") increases with length. Add that test to the pre-registered analysis.

**"Missing information" rule.** 12.7% of positive "met" verdicts (1,439 / 11,345) and 13.7% of negative "met" verdicts (102 / 747) have an explanation that mentions missing information (keyword flag, so an upper bound). By category: surgery 17.7%, radiotherapy 16.2%, systemic therapy 13.2%, complementary exams 5.4%. A seeded sample of 60 flagged verdicts is in `validation/output/missing_info_spotcheck.csv` (gitignored, contains case text) for a manual check of whether the exception was applied correctly.

## C and D. Cross-provider judges and repeated grading (need API budget)

Scripts are ready. `evals/` and `.env` are gitignored; when running from another checkout or worktree, set `BCA_DATA_ROOT` to the checkout that has them.

- `validation/regrade.py --judge <backend>:<model> [--cases-per-stage N] [--repeats K] [--dry-run]` re-grades with byte-identical prompts, resumable, output in `validation/output/regrades/` (gitignored).
- `validation/judge_agreement.py` → `validation/results/judge_agreement.json`:
  - C: κ, Gwet's AC1 and % agreement vs the original judge (overall, by sign, by category; cluster-bootstrap CI), case-score correlation, configuration-ranking Kendall's τ and per-configuration paired-bootstrap Δ, and self-preference (difference in differences: Gemini-generated minus other responses, original judge vs other judge).
  - D: flip rate, Fleiss' κ, SD of case and configuration scores across repeats vs the bootstrap SD over cases, and ranking τ between repeats. For `gemini-2.5-flash` with default settings the original grading counts as one repeat.
  - The agreement functions are reused for the clinician comparison (B).
- Tested end to end on synthetic re-grades (the original verdicts with injected flips); the statistics recovered the injected disagreement.

**Size of a run** (input tokens from the dry run, calibrated on the original grading):

| Subset | Calls per pass | Input tokens per pass |
|---|---|---|
| All 72 cases × 27 configurations | 30,969 | ≈ 78M |
| 6 cases per stage (24 cases) × 27 configurations | 10,179 | ≈ 25M |

Plus about 150 output tokens per call and any reasoning tokens (≈ 520 per call for Gemini 2.5 Flash). The response sits before the criterion in the prompt, so prompt caching (automatic on OpenAI, `cache_control` on Anthropic) could cut input cost a lot, since every criterion of a response shares the same prefix.

**Suggested plan** (decision pending on budget, open question in #11):
1. D: `gemini:gemini-2.5-flash`, 24-case subset, `--repeats 2` (with the original, k = 3). Cheapest, and it directly measures instability of the judge that produced the results.
2. C: `litellm:gpt-5.5-2026-04-23` and `litellm:anthropic/claude-opus-4-8` on the same 24 cases, one pass each; optionally `gemini:gemini-3.1-pro-preview` as a stronger Gemini. Note: GPT-5.5 and Claude Opus 4.8 are also evaluated models, so they have their own possible self-preference; the analysis reports the self-preference contrast for the Gemini judge.
3. If agreement is high, stop. If not, extend C to all 72 cases.

Needs `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` in `.env` (only `GEMINI_API_KEY` is there now).

## Status vs "Done when" in #11
- [x] Judge identity, settings and what the judge sees: confirmed from code and data (above)
- [ ] Protocol text moved into Methods and the Supplement (draft above)
- [ ] D and C runs (budget decision)
- [ ] B: judge–clinician agreement (needs #14)
- [ ] Response-to-reviewers text
