"""Shared helpers for the clinician review study (issues #13 and #14).

Everything these scripts write goes under ``validation/output/``, which is
gitignored: the outputs contain patient case data and must never be committed.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVALS_DIR = REPO_ROOT / "evals"
EVAL_DATASET = REPO_ROOT / "data" / "eval_dataset.jsonl"
OUTPUT_DIR = REPO_ROOT / "validation" / "output"

SEED = 2026

# Cases flagged in #12 as possible stage 0 / DCIS, which conflicts with the
# stated eligibility (invasive, stages I-IV). Kept out of the review sample
# until that question is resolved.
STAGE0_PATTERN = re.compile(r"Estadiamento cl[ií]nico:[^\n]*\bTis\b", re.IGNORECASE)


def response_text(agent_response) -> str:
    """Multi-agent pipelines store the final answer as {'text': ...}."""
    if isinstance(agent_response, dict):
        return agent_response.get("text", "")
    return agent_response or ""


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def _maybe_literal(value):
    """Eval files store some fields as Python reprs; parse them back."""
    if isinstance(value, str) and value[:1] in "[{":
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def load_cases() -> dict[str, dict]:
    """Load the evaluation dataset keyed by prompt_id (as str)."""
    cases = {}
    with open(EVAL_DATASET, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            cases[str(row["prompt_id"])] = row
    return cases


def case_score(rubrics: list[dict], grader_responses: list[dict]) -> float:
    """HealthBench case score (unclipped): met points / total positive points."""
    positive = sum(r["points"] for r in rubrics if r["points"] > 0)
    achieved = sum(
        r["points"] for r, g in zip(rubrics, grader_responses) if g.get("criteria_met")
    )
    return achieved / positive if positive else float("nan")


def load_evaluated_responses() -> list[dict]:
    """Load every evaluated (configuration, case) pair from ``evals/``.

    Returns one dict per response with the configuration, the response text,
    the per-criterion grader verdicts and the unclipped case score.
    """
    rows = []
    for path in sorted(EVALS_DIR.glob("*/*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        pipeline = data.get("pipeline_id") or path.parent.name
        model = data.get("model_id") or path.stem
        for ex in data["per_example_details"]:
            rubrics = _maybe_literal(ex["rubrics"])
            grader = _maybe_literal(ex["grader_responses"])
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


PATIENT_ID_LINE = re.compile(r"^\s*-?\s*ID da paciente:.*$", re.MULTILINE | re.IGNORECASE)


def deidentify_summary(text: str) -> str:
    """Drop the EHR patient-ID line; reviewers see a study case code instead."""
    return PATIENT_ID_LINE.sub("", text).strip("\n")
