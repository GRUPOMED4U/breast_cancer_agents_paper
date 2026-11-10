"""Shared helpers for the LLM-as-a-judge validation scripts (#11).

Self-contained on purpose, so this work does not depend on the clinician
review-material scripts (#13).

Data location: ``evals/`` and ``.env`` are gitignored, so a fresh checkout or
worktree does not have them. Set ``BCA_DATA_ROOT`` to the checkout that holds
them (default: this repository).

Outputs that contain case text or judge explanations go under
``validation/output/``, which is made self-ignoring (it gets its own
``.gitignore`` with ``*``) so they can never be committed.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("BCA_DATA_ROOT", REPO_ROOT)).resolve()
EVALS_DIR = DATA_ROOT / "evals"
OUTPUT_DIR = REPO_ROOT / "validation" / "output"
RESULTS_DIR = REPO_ROOT / "validation" / "results"

SEED = 2026

dotenv.load_dotenv(DATA_ROOT / ".env")


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ignore = OUTPUT_DIR / ".gitignore"
    if not ignore.exists():
        ignore.write_text("# Patient case data: never commit\n*\n", encoding="utf-8")
    return OUTPUT_DIR


def response_text(agent_response) -> str:
    """Multi-agent pipelines store the final answer as {'text': ...}."""
    if isinstance(agent_response, dict):
        return agent_response.get("text", "")
    return agent_response or ""


def maybe_literal(value):
    """Eval files store some fields as Python reprs; parse them back."""
    if isinstance(value, str) and value[:1] in "[{":
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def case_score(rubrics: list[dict], grader_responses: list[dict]) -> float:
    """HealthBench case score (unclipped): met points / total positive points."""
    positive = sum(r["points"] for r in rubrics if r["points"] > 0)
    achieved = sum(r["points"] for r, g in zip(rubrics, grader_responses) if g.get("criteria_met"))
    return achieved / positive if positive else float("nan")


def load_evaluated_responses() -> list[dict]:
    """One dict per graded (configuration, case) response in ``evals/``."""
    rows = []
    for path in sorted(EVALS_DIR.glob("*/*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        pipeline = data.get("pipeline_id") or path.parent.name
        model = data.get("model_id") or path.stem
        for ex in data["per_example_details"]:
            rubrics = maybe_literal(ex["rubrics"])
            grader = maybe_literal(ex["grader_responses"])
            rows.append(
                {
                    "config": f"{pipeline}::{model}",
                    "pipeline": pipeline,
                    "model": model,
                    "eval_model": data.get("eval_model_id"),
                    "prompt_id": str(ex["prompt_id"]),
                    "agent_response": ex["agent_response"],
                    "rubrics": rubrics,
                    "grader_responses": grader,
                    "score": case_score(rubrics, grader),
                }
            )
    return rows
