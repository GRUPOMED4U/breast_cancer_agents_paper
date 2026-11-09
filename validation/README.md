# Clinician review study: materials and descriptives

Code for issue #13, which prepares the materials for the clinician review study in #14 and the AIRG transparency analyses (option F in #10).

> **Patient data never goes to GitHub.** Every script writes to `validation/output/`, which is gitignored, and `*.xlsx` files are ignored repo-wide. Only the code in this folder is committed. Send workbooks to reviewers through an approved private channel, never through the repository.

## Pipeline

Run from the repository root, in this order:

```bash
.venv/Scripts/python.exe validation/sample_cases.py        # allocation.csv
.venv/Scripts/python.exe validation/select_responses.py    # selected_responses.jsonl, answer_key_responses.csv, blinding_review.tsv
.venv/Scripts/python.exe validation/build_workbooks.py     # workbooks/*.xlsx, answer_key_rubrics.csv
.venv/Scripts/python.exe validation/rubric_descriptives.py # descriptives/
```

Everything is seeded (`SEED = 2026` in `common.py`), so re-running gives the same sample, the same responses and the same workbooks.

## Design choices (see #14)

- **Allocation:** in each stage, 3 cases are shared by both oncologists, 3 go only to oncologist 1 and 3 only to oncologist 2. That gives 36 distinct cases and 24 per oncologist. Two calibration cases are drawn from outside the sample. Stage 0 / DCIS cases (prompt_ids 5, 7, 8) are excluded until #12 is resolved. `--per-group 2` gives the reduced design.
- **Primary response** (holistic rating, met / not met for every criterion): primary responses are balanced over within-case score terciles and spread over the 27 configurations, with at most 2 uses per configuration. Shared cases use the same response for both oncologists.
- **Top-up response** (optional step 5, negative criteria only): enriched for responses where the judge marked a negative criterion as met. **Analyse top-up items separately**, because the enrichment changes how often criteria are met.
- **Blinding:** the EHR patient-ID line is removed from summaries. Patient IDs in responses, including mis-copied ones, are replaced with the case code. Phrases that reveal the pipeline (specialist agents, fact checking) and revision preambles become `[removido]`. Responses that are refusals or shorter than 150 words are not eligible. **Remaining limitation:** responses are in English or Portuguese depending on the configuration (e.g., Claude zero-shot is almost always English, GPT-5.5 always Portuguese), so language can partially unblind the model.
- **Workbooks:** one sheet per case. Steps 2, 3 and 5 are collapsed row groups. Hidden column L holds a stable item id on each answer row, which is what the analysis reads back. Examples: `holistic_rating`, `ref_class`, `rubric:3` (the answers for criterion 3 are in columns E–K, in the order of `RUBRIC_COLS` in `build_workbooks.py`), and `topup:3`. The case code is in L1.

## Before sending the workbooks

1. Open `validation/output/blinding_review.tsv` and check every remaining flagged span.
2. Open one workbook in Excel and check the layout and drop-downs. Pilot one case and time it (task 3 in #13).
3. Read the calibration workbook (`calibracao.xlsx`) yourself before the training session.
