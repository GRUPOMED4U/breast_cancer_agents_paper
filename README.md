# Agentic systems for breast cancer treatment recommendations

Source code and experiment configuration for the paper *"Agentic systems for breast
cancer treatment recommendations"*.

The study benchmarks agentic LLM systems on the task of generating breast cancer
treatment recommendations. It evaluates **seven pipelines** (single-LLM baselines,
tool-augmented single LLMs, and multi-agent architectures) against **72 real clinical
cases** (18 per stage, I–IV) using **1,147 case-specific rubrics** produced through
**Asymmetric Information Rubric Generation (AIRG)** — a workflow in which the rubric
generator has privileged access to the real clinical decisions that the evaluated
models never see. Scoring follows the rubric-based, example-level procedure introduced
in HealthBench.

> **Note on data.** The clinical dataset is derived from real patient records from a
> private oncology clinic (IRB protocol CAAE 90422925.6.0000.0096) and is **not**
> included in this repository. This repo contains the code, pipeline definitions, and experiment
> configuration needed to reproduce the methodology.

## Pipeline overview

| ID (pipeline key) | Paper name | Description |
|---|---|---|
| `single_llm_zero_shot` | Baseline | Single LLM, no tools |
| `single_llm_with_web_search` | WS | Single LLM + web search (Parallel.ai) |
| `single_llm_with_pubmed_search` | PM | Single LLM + PubMed search |
| `single_llm_with_pubmed_search_and_full_pmc_articles` | PM+PMC | Single LLM + PubMed + full-text PMC retrieval |
| `divide_and_conquer` | D&C | Orchestrator + 4 topic subagents (systemic therapy, radiotherapy, surgery, complementary exams), all with tools |
| `divide_and_conquer_with_fact_checker` | D&C+FC | D&C with a handoff to a fact-checker agent |
| `divide_and_conquer_with_subagents_auto_spawning` | D&C+SA | Orchestrator autonomously spawns custom subagents, then a fact checker |

Pipelines are agent systems built on the **OpenAI Agents SDK** (via LiteLLM, so Google,
OpenAI, and Anthropic models are all supported). Tools are exposed either as **MCP servers** or **functional tools**.

## Repository layout

### Data pipeline (orchestrated with [DVC](https://dvc.org))

The end-to-end workflow — prepare case summaries → generate rubrics → run inference — is
defined in [`dvc.yaml`](dvc.yaml) and reproducible with `dvc repro`.

| Path | Role |
|---|---|
| [`prepare_case_summaries.py`](prepare_case_summaries.py) | Builds structured case summaries from the raw per-stage case CSVs → `data/case_summaries.jsonl` |
| [`generate_rubrics.py`](generate_rubrics.py) | AIRG rubric generation. Runs the rubric-generator agent (with reference decisions + search tools) → `data/eval_dataset.jsonl` |
| [`dataset.py`](dataset.py) | `Dataset` — list-like loader/merger for JSONL evaluation entries |
| [`data_models.py`](data_models.py) | Pydantic schemas: `Rubric`, `RubricTag`, `EvaluationEntry`, `EvaluationMetrics`, etc. |
| `data/` | Input cases (`cases_stage_{1..4}.csv`), `case_summaries.jsonl`, `eval_dataset.jsonl`. Git-ignored except schema. |

### Inference (running the pipelines)

| Path | Role |
|---|---|
| [`run_inference.py`](run_inference.py) | CLI entry point: runs one `--pipeline` with one `--model` over the eval dataset; writes to `results/` |
| [`pipelines/`](pipelines) | One module per pipeline; [`base_class.py`](pipelines/base_class.py) defines the interface and [`__init__.py`](pipelines/__init__.py) exposes `PIPELINE_REGISTRY` |
| [`experiments/`](experiments) | One YAML per pipeline. Each holds a list of experiments (one per model) with agent instructions, model settings, concurrency limits, and prompt templates |
| [`agent_definitions.yaml`](agent_definitions.yaml) | Agent definition for the rubric generator (OpenAI Agents SDK format) |
| `results/` | Inference outputs, organized as `results/<pipeline>/<model_id>.jsonl` |

### Tools & MCP servers

| Path | Role |
|---|---|
| [`custom_mcps/pubmed_search/`](custom_mcps/pubmed_search) | PubMed search MCP server (`server.py`) and NCBI API client (`pubmed_client.py`) — search articles, fetch abstracts, retrieve full-text PMC manuscripts |
| [`custom_tools/parallel_ai.py`](custom_tools/parallel_ai.py) | Web search / web fetch tool backed by Parallel.ai |
| [`mcp_utils.py`](mcp_utils.py) | Helpers for wiring MCP servers into agents |
| [`custom_error_handlers.py`](custom_error_handlers.py) | Error/retry handling for agent runs |
| [`prompt_helpers.py`](prompt_helpers.py) | Prompt construction helpers |

### Evaluation

| Path | Role |
|---|---|
| [`run_eval.py`](run_eval.py) | CLI entry point: grades inference results for a `--pipeline`/`--model` against the rubrics; writes to `evals/` |
| [`evaluator.py`](evaluator.py) | `AsyncGeminiEvaluator` — LLM-as-a-judge grader implementing the HealthBench example-level scoring, category-specific scores, and bootstrap uncertainty |
| `evals/` | Evaluation metrics, as `evals/<pipeline>/<model_id>.json`, plus stored `agent_responses/` |

### Analysis, plots & reporting

| Path | Role |
|---|---|
| [`plots.py`](plots.py) | Generates all figures in the paper |
| [`assets/`](assets) | Rendered figures (`.png`) and generated tables (`.tex`) used in the paper |


### Project configuration

| Path | Role |
|---|---|
| [`pyproject.toml`](pyproject.toml) / `uv.lock` | Dependencies (managed with [uv](https://docs.astral.sh/uv/)); Python ≥ 3.12 |
| [`dvc.yaml`](dvc.yaml) / `dvc.lock` / `.dvc/` | DVC pipeline definition and cache |
| `.env` | API keys (see below) — not committed |
| [`logs.py`](logs.py) | Logging setup |

## Setup

Requires Python ≥ 3.12. Dependencies are managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Create a `.env` file with the required API keys:

```
GEMINI_API_KEY=...
PARALLEL_API_KEY=...     # web search tool
NCBI_API_KEY=...         # PubMed search
NCBI_EMAIL=...           # PubMed / NCBI contact email
# plus provider keys for any OpenAI / Anthropic models you run
```

## Usage

Run the full data + inference pipeline via DVC:

```bash
dvc repro
```

Or run steps individually.

**Run inference** for one pipeline/model combination:

```bash
python run_inference.py \
  --pipeline divide_and_conquer_with_subagents_auto_spawning \
  --model anthropic/claude-opus-4-8 \
  --save-steps 5
```

**Evaluate** the generated recommendations against the rubrics:

```bash
python run_eval.py \
  --pipeline divide_and_conquer_with_subagents_auto_spawning \
  --model anthropic/claude-opus-4-8 \
  --grader-model gemini-2.5-flash
```

Available pipeline keys are listed in the table above (and in
[`pipelines/__init__.py`](pipelines/__init__.py)). The models evaluated in the paper were
`gemini-2.5-flash`, `gemini-3.1-pro-preview`, `gemini-3.5-flash`, `gpt-5.5-2026-04-23`,
and `claude-opus-4-8`; the specific models configured per pipeline live in the
corresponding `experiments/<pipeline>.yaml`.

## Method summary

1. **Data collection** — 72 histologically confirmed invasive breast cancer cases
   (2023–2025) reviewed by medical students under physician supervision; structured into
   case summaries plus the real clinical decisions made for each patient.
2. **AIRG rubric generation** — a frontier LLM (high reasoning effort) receives the case
   summary, the *reference clinical decisions*, and web/PubMed search tools, and produces
   case-specific rubrics grouped into four categories: systemic therapy, radiotherapy,
   surgical therapy, and complementary exams. The evaluated models see only the case
   summary — this information asymmetry grounds the rubrics without full expert authoring.
3. **Inference** — each pipeline × model generates treatment recommendations.
4. **Scoring** — an LLM-as-a-judge grades each response criterion-by-criterion; the
   example-level score is the earned points divided by the maximum attainable positive
   points, aggregated (and clipped to [0, 1]) across cases, with bootstrap uncertainty.
   Scores are also computed per rubric category and per disease stage.
5. **Error analysis** — a certified oncologist manually reviewed a sample of top-model
   responses to characterize clinically relevant failure modes.

## Citation

Please cite the associated paper if you use this code. (Citation details to be added.)
