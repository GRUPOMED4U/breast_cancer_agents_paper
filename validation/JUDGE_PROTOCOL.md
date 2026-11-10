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

## C and D. Cross-provider judges and repeated grading

D and C have been run (results below): D with Gemini 2.5 Flash repeats, C with GPT-5.4 mini and Claude Sonnet 5.5. `evals/` and `.env` are gitignored; when running from another checkout or worktree, set `BCA_DATA_ROOT` to the checkout that has them.

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

### D results: judge stability (run 9 Nov 2026)

Gemini 2.5 Flash, default settings (same as the original grading), 24 cases (6 per stage, seeded: 3, 9, 10, 14, 15, 17, 27, 30, 31, 32, 33, 35, 36, 37, 43, 45, 49, 50, 54, 57, 63, 64, 67, 68) × 27 configurations, 2 new repeats + the original = **3 gradings of 10,179 (response, criterion) pairs**. 20,358 calls, 0 failures; the API reported the served model as `gemini-2.5-flash`. Output: `validation/results/judge_agreement.json`.

| Measure | Value |
|---|---|
| Pairs whose verdict differs in at least one of 3 gradings (flip rate) | **8.6%** (positive criteria 7.0%, negative 12.8%) |
| Fleiss' κ across the 3 gradings | **0.88** |
| Pairwise Cohen's κ | 0.88 / 0.88 / 0.88 |
| New repeat vs original: % agreement, κ (95% CI), AC1 | 94.2%, 0.88 (0.85–0.90), 0.89 |
| Same, positive criteria | 95.3%, κ 0.91, AC1 0.91 |
| Same, negative criteria | 91.2%, κ 0.52, AC1 0.89 |
| Same, by category (κ) | complementary exams 0.91, radiotherapy 0.90, systemic therapy 0.87, surgery 0.80 |
| Met-rate, original vs repeat | 39.9% vs 39.8% (no systematic drift) |
| Case score: SD across gradings | mean 0.058, max 0.26 |
| Case score: original vs repeat | Pearson r 0.86, Spearman ρ 0.85 |
| Configuration score: SD across gradings | 0.001–0.036 (median 0.39 × the bootstrap SD over cases, which is 0.029–0.065) |
| Configuration ranking: Kendall's τ between gradings | 0.79–0.85 |
| Largest change in a configuration's rank (out of 27) | 8 places |

Reading:
- Criterion-level verdicts are highly but not perfectly repeatable (κ ≈ 0.88). One in 12 verdicts can change on a re-run.
- Negative criteria are less stable: low κ (0.52) with high AC1 (0.89) is the usual low-prevalence pattern (only ~8% of negative criteria are met), but the flip rate is also higher (12.8%). The judge is least consistent on exactly the items that subtract points.
- Surgery criteria are the least stable category (κ 0.80).
- The noise mostly averages out. Single case scores move noticeably (mean SD 0.06), but configuration scores move by less than half of the sampling uncertainty already reported (median ratio 0.39). Adding judge noise to the case-sampling variance (0.39² ≈ 0.15 of it) widens a configuration score's SD by only about 7% (√1.15 ≈ 1.07), on this 24-case subset.
- Conclusions that hold in all 3 gradings: the best configuration (D&C+FC with Gemini 3.5 Flash, rank 1 in all 3), D&C+SA as the best pipeline on average, and zero-shot as the worst. The order of the pipelines in the middle (D&C, D&C+FC, WS, PM, PM+PMC; average scores within ~0.03 of each other) changes between gradings, so the paper should not claim differences among them without accounting for judge noise.
- As expected for the same judge, there is no self-preference contrast (difference in differences −0.002, CI −0.024 to 0.020). This also checks that the self-preference statistic reads zero when it should.

Proposed text: "To quantify judge instability, we re-graded a stratified subset of 24 cases (all 27 configurations; 10,179 criterion judgments) twice more with the same judge and settings. Verdicts agreed across the three gradings with Fleiss' κ = 0.88 (8.6% of judgments changed at least once; 12.8% for negative criteria). Judge variability in configuration-level scores was small relative to case-sampling uncertainty (median SD ratio 0.39), and the top configuration and the best and worst pipelines were unchanged across gradings, whereas the ordering of the intermediate pipelines was not stable."

### C results: cross-provider judge, GPT-5.4 mini (run 10 Nov 2026)

**Choice of judges.** Cross-provider judges were chosen at the same tier as the original judge (Gemini 2.5 Flash is a mid-tier "Flash" model) and from models that were **not** evaluated in the paper, so they have no responses of their own to favor. GPT-5.4 mini first; Claude Sonnet 5.5 next. Frontier judges (GPT-5.5, Claude Opus 4.8) were set aside on cost and because both are also evaluated models.

**Setup.** `gpt-5.4-mini-2026-03-17` (OpenAI API, the only snapshot of that model on the account), `reasoning_effort=medium` (the model does not reason by default; the original Gemini judge reasoned on every call, so reasoning was turned on to match), provider defaults otherwise, same byte-identical prompts, same 24 cases × 27 configurations as D, one pass. 10,179 calls, 0 failures, every call reasoned (mean 117 reasoning tokens), total cost $24.65 ($0.0024 per call). A first attempt with the provider default (no reasoning) was stopped after 203 calls and discarded, so all verdicts share one configuration.

| Measure (GPT-5.4 mini vs original Gemini 2.5 Flash) | Value |
|---|---|
| % agreement, Cohen's κ (95% CI), Gwet's AC1 | 91.3%, **0.81** (0.78–0.85), 0.84 |
| For comparison: Gemini vs itself (D) | 94.2%, κ 0.88 |
| Positive criteria | 91.1%, κ 0.82, AC1 0.82 |
| Negative criteria | 91.9%, κ 0.51, AC1 0.90 |
| By category (κ) | radiotherapy 0.87, complementary exams 0.86, systemic therapy 0.79, surgery 0.71 |
| Met-rate | Gemini 39.9% vs GPT 33.6% |
| Direction of disagreements | Gemini "met" / GPT "not met": 762; the reverse: 121 |
| Case score | Pearson r 0.82, Spearman ρ 0.81; mean 0.49 (Gemini) vs 0.43 (GPT) |
| Configuration score difference (GPT − Gemini) | mean −0.064, range −0.12 to −0.02 |
| Configuration ranking, Kendall's τ | **0.87** (largest rank change: 5 places out of 27) |
| Self-preference (difference in differences) | −0.025 (−0.050 to −0.001) |

Reading:
- **Agreement across providers is close to the judge's agreement with itself** (κ 0.81 vs 0.88), so most of the cross-judge disagreement is of the same size as the run-to-run noise measured in D.
- **GPT-5.4 mini is stricter.** Almost all disagreements go one way (Gemini says met, GPT says not met: 6:1), so absolute scores are about 0.06 lower under GPT. Absolute scores therefore depend on the judge and should be presented as judge-specific; comparisons between configurations are what transfer.
- **The ranking holds.** The top 3 configurations are the same and in the same order under both judges (D&C+FC with Gemini 3.5 Flash, then D&C+SA with Gemini 3.5 Flash, then D&C+SA with Claude Opus 4.8). D&C+SA is the best pipeline on average (0.544 Gemini / 0.469 GPT) and zero-shot the worst (0.460 / 0.405) under both. As in D, the intermediate pipelines (D&C, D&C+FC, WS, PM, PM+PMC; within ~0.02 of each other under GPT) change order.
- **No evidence that the Gemini judge favors Gemini responses.** The gap between Gemini-generated and other responses is −0.062 under the Gemini judge and −0.037 under GPT; the difference (−0.025) points the other way, i.e. the Gemini judge is, if anything, slightly harder on Gemini responses relative to GPT-5.4 mini. GPT-5.4 mini is not an evaluated model, but it may favor responses from GPT-5.5 (part of the "other" group), so this contrast is read only as "no Gemini self-preference", not as GPT bias.
- Surgery is again the least reliable category, as in D.

Proposed text: "To test whether the results depend on the judge, we re-graded the same 24-case subset with a judge from another provider, GPT-5.4 mini (OpenAI; reasoning effort medium), which was not among the evaluated models. Criterion-level agreement with the original judge was κ = 0.81 (91.3% agreement), close to the original judge's agreement with itself (κ = 0.88). GPT-5.4 mini was stricter (met-rate 33.6% vs 39.9%), lowering absolute scores by 0.064 on average, but the configuration ranking was preserved (Kendall's τ = 0.87), with the same top three configurations and the same best (D&C+SA) and worst (zero-shot) pipelines. We found no evidence that the Gemini judge favored Gemini-generated responses."

### C results: cross-provider judge, Claude Sonnet 5.5 (run 10 Nov 2026)

**Setup.** `claude-sonnet-5-5` (Anthropic API), provider defaults: adaptive reasoning, which is always on for this model, at its default effort; `max_tokens` 16,000. Same prompt text; for Anthropic it is sent as two consecutive text blocks split at `# Rubric`, so the shared prefix (instructions, case, response) can be cached (75% of input tokens were cache reads). Same 24 cases × 27 configurations, one pass. 10,179 calls, 0 failures, total cost $47.27. Claude wrote its explanations in Portuguese (Gemini in English); this does not affect the met / not met verdicts.

| Measure | Sonnet 5.5 vs Gemini | GPT-5.4 mini vs Gemini | Sonnet 5.5 vs GPT-5.4 mini |
|---|---|---|---|
| % agreement, Cohen's κ | 92.1%, **0.83** (0.79–0.86) | 91.3%, 0.81 | 93.8%, **0.86** |
| Gwet's AC1 | 0.85 | 0.84 | 0.89 |
| Positive criteria: κ | 0.85 | 0.82 | |
| Negative criteria: κ / AC1 | **0.28** / 0.89 | 0.51 / 0.90 | |
| Met-rate, positive criteria | 47.3% vs 50.7% | 43.2% vs 50.7% | |
| Met-rate, negative criteria | **3.5%** vs 10.6% | 7.7% vs 10.6% | |
| Case score: Pearson r | 0.78 | 0.82 | |
| Mean case score (Gemini 0.490) | 0.493 | 0.426 | |
| Configuration ranking: Kendall's τ | 0.79 (largest rank change 7) | 0.87 (5) | 0.81 |
| Self-preference (difference in differences) | 0.000 (−0.018 to 0.020) | −0.025 (−0.050 to −0.001) | |

By category (κ vs Gemini): complementary exams 0.86, radiotherapy 0.86, systemic therapy 0.81, surgery 0.76.

**Pipeline averages under each judge** (mean of the configuration scores in each pipeline, 24-case subset):

| Judge | Order |
|---|---|
| Gemini 2.5 Flash (original) | D&C+SA 0.544 > D&C+FC 0.494 > D&C 0.487 > WS 0.486 > PM 0.481 > PM+PMC 0.466 > Zero-shot 0.460 |
| GPT-5.4 mini | D&C+SA 0.469 > WS 0.426 > PM 0.424 > D&C 0.416 > D&C+FC 0.415 > PM+PMC 0.413 > Zero-shot 0.405 |
| Claude Sonnet 5.5 | D&C+SA 0.537 > WS 0.489 > PM 0.487 > Zero-shot 0.486 > D&C+FC 0.484 > D&C 0.479 > PM+PMC 0.468 |

Reading, across the three judges (and D):
- **Criterion-level agreement is high for every pair of judges** (κ 0.81–0.86), and close to Gemini's agreement with itself (0.88). The two new judges agree with each other more (0.86) than either does with Gemini, so on the margin Gemini is the outlier, mainly by being more lenient on positive criteria.
- **Negative criteria are where judges differ most.** Sonnet 5.5 marks a negative criterion as met in 3.5% of cases, Gemini in 10.6%, GPT-5.4 mini in 7.7%. Because negative criteria subtract points, Sonnet's stricter positives and more lenient negatives cancel out, so its mean score (0.493) matches Gemini's (0.490). Which judge is right on negative criteria cannot be settled from the judges alone: **this is the key question for the clinician labels (B)**, which should report judge–clinician agreement separately for negative criteria.
- **Robust across all three judges:** D&C+SA is the best pipeline on average, by a clear margin (0.04–0.05 above the next pipeline under every judge), and the same three configurations make up the top 3 (D&C+FC with Gemini 3.5 Flash, D&C+SA with Gemini 3.5 Flash, D&C+SA with Claude Opus 4.8), although their order changes under Sonnet.
- **Not robust:** the order of the other six pipelines. Under each judge they lie within ~0.02–0.03 of each other, and the order changes between judges. In particular, **zero-shot is the worst pipeline under Gemini and GPT, but fourth under Sonnet**. The paper should not claim that the tool-augmented single-LLM pipelines, D&C or D&C+FC beat zero-shot.
- **No self-preference:** neither cross-judge comparison suggests the Gemini judge favors Gemini-generated responses (Sonnet: 0.000; GPT: −0.025).
- **Surgery is the least reliable category** under every comparison (κ 0.71–0.80), as in D.

Proposed text: "To test whether the results depend on the judge, we re-graded a stratified 24-case subset (all 27 configurations; 10,179 criterion judgments) with two judges from other providers that were not among the evaluated models, GPT-5.4 mini (OpenAI) and Claude Sonnet 5.5 (Anthropic), both with reasoning enabled. Criterion-level agreement with the original judge was κ = 0.81 and 0.83 (κ = 0.86 between the two new judges), close to the original judge's agreement with itself on repeated grading (κ = 0.88). Agreement was lowest on negative (penalty) criteria, where the judges differed in how often a penalty applied (3.5%–10.6%). Configuration rankings were largely preserved (Kendall's τ = 0.87 and 0.79). Under all judges, D&C+SA was the best pipeline and the same three configurations ranked highest, whereas the ordering of the remaining pipelines, including zero-shot, depended on the judge. We found no evidence that the Gemini judge favored Gemini-generated responses."

## Status vs "Done when" in #11
- [x] Judge identity, settings and what the judge sees: confirmed from code and data (above)
- [ ] Protocol text moved into Methods and the Supplement (draft above)
- [x] D: judge stability (24-case subset, 3 gradings)
- [x] C, GPT-5.4 mini (24-case subset)
- [x] C, Claude Sonnet 5.5 (24-case subset)
- [ ] B: judge–clinician agreement (needs #14)
- [ ] Response-to-reviewers text
