"""Re-grade stored responses with other judges or repeated runs (#11, options C and D).

Every (response, criterion) pair is graded with the *same* prompt the original
judge received (``evaluator.build_grader_prompt``, rebuilt from ``evals/``;
checked to be byte-identical), one call per pair, so only the judge changes.

Judges are given as ``<backend>:<model>``:
  gemini:gemini-2.5-flash          google-genai, same request config as evaluator.py
  gemini:gemini-3.1-pro-preview    (stronger Gemini)
  litellm:gpt-5.5-2026-04-23       any LiteLLM model id (needs the provider key in .env)
  litellm:anthropic/claude-opus-4-8

Subsets: by default all 72 cases x 27 configurations (~31k calls per judge per
repeat). ``--cases-per-stage N`` takes a seeded, stage-stratified sample of
cases; ``--configs`` restricts configurations (``pipeline::model``).

Repeats (option D): ``--repeats K`` writes rep1..repK. Re-running is resumable:
pairs already in a file are skipped. Run with ``--dry-run`` first: it prints the
number of calls and an input-token estimate and makes no API calls.

Output (gitignored, contains judge explanations of patient cases):
  validation/output/regrades/<judge_slug>/rep<k>.jsonl
  one line per verdict: prompt_id, config, criterion_idx, criteria_met (bool or
  null on failure), explanation, model_version, usage, error.

Usage:
    .venv/Scripts/python.exe validation/regrade.py --judge litellm:gpt-5.5-2026-04-23 --cases-per-stage 6 --dry-run
    .venv/Scripts/python.exe validation/regrade.py --judge gemini:gemini-2.5-flash --cases-per-stage 6 --repeats 3
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import random
import re
import sys
from pathlib import Path

from tqdm.asyncio import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_models import EvaluationEntry, GraderResponse  # noqa: E402
from evaluator import build_grader_prompt  # noqa: E402

from judge_common import EVALS_DIR, OUTPUT_DIR, SEED, ensure_output_dir, maybe_literal  # noqa: E402


CHARS_PER_TOKEN = 4.0  # calibrated on the original grading (78.0M prompt tokens for 30,969 calls)
MAX_ATTEMPTS = 4
JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def judge_slug(judge: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", judge.split(":", 1)[1])


def load_entries() -> list[tuple[str, EvaluationEntry]]:
    """(config, EvaluationEntry) for every graded response in evals/."""
    out = []
    for path in sorted(EVALS_DIR.glob("*/*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        config = f"{data.get('pipeline_id') or path.parent.name}::{data.get('model_id') or path.stem}"
        for ex in data["per_example_details"]:
            ex = {k: maybe_literal(v) if k in ("rubrics", "grader_responses", "prompt", "agent_response") else v for k, v in ex.items()}
            out.append((config, EvaluationEntry(**ex)))
    return out


def sample_cases(entries, per_stage: int | None, seed: int) -> set[str] | None:
    if per_stage is None:
        return None
    by_stage = collections.defaultdict(set)
    for _, e in entries:
        by_stage[str(getattr(e, "breast_cancer_stage", ""))].add(e.prompt_id)
    rng = random.Random(seed)
    chosen = set()
    for stage in sorted(by_stage):
        ids = sorted(by_stage[stage], key=int)
        chosen.update(rng.sample(ids, min(per_stage, len(ids))))
    return chosen


def parse_verdict(text: str) -> GraderResponse:
    try:
        return GraderResponse.model_validate_json(text)
    except Exception:
        m = JSON_OBJ.search(text or "")
        if not m:
            raise
        return GraderResponse.model_validate_json(m.group(0))


class Judge:
    def __init__(self, spec: str, temperature: float | None, seed: int | None):
        self.backend, self.model = spec.split(":", 1)
        self.temperature, self.seed = temperature, seed
        if self.backend == "gemini":
            from google import genai

            self.client = genai.Client()
        elif self.backend == "litellm":
            import litellm

            litellm.suppress_debug_info = True
            self.litellm = litellm
        else:
            raise SystemExit(f"Unknown backend '{self.backend}' (use gemini: or litellm:)")

    async def grade(self, prompt: str) -> dict:
        if self.backend == "gemini":
            config = {
                "response_mime_type": "application/json",
                "response_json_schema": GraderResponse.model_json_schema(),
            }
            if self.temperature is not None:
                config["temperature"] = self.temperature
            if self.seed is not None:
                config["seed"] = self.seed
            r = await self.client.aio.models.generate_content(model=self.model, contents=prompt, config=config)
            usage = r.usage_metadata.model_dump(exclude_none=True) if r.usage_metadata else {}
            return {"text": r.text, "model_version": r.model_version, "usage": usage}
        kwargs = {}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.seed is not None:
            kwargs["seed"] = self.seed
        r = await self.litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            drop_params=True,
            **kwargs,
        )
        usage = r.usage.model_dump() if getattr(r, "usage", None) else {}
        return {"text": r.choices[0].message.content, "model_version": r.model, "usage": usage}


async def run_rep(judge: Judge, jobs: list[dict], out_path: Path, concurrency: int) -> None:
    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("criteria_met") is not None:
                    done.add((rec["prompt_id"], rec["config"], rec["criterion_idx"]))
    todo = [j for j in jobs if (j["prompt_id"], j["config"], j["criterion_idx"]) not in done]
    print(f"{out_path}: {len(done)} done, {len(todo)} to grade")
    if not todo:
        return

    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "a", encoding="utf-8") as f:

        async def one(job):
            rec = {k: job[k] for k in ("prompt_id", "config", "criterion_idx")}
            rec.update(criteria_met=None, explanation="", model_version=None, usage={}, error=None)
            async with sem:
                for attempt in range(MAX_ATTEMPTS):
                    try:
                        raw = await judge.grade(job["prompt"])
                        verdict = parse_verdict(raw["text"])
                        rec.update(
                            criteria_met=verdict.criteria_met,
                            explanation=verdict.explanation,
                            model_version=raw["model_version"],
                            usage=raw["usage"],
                            error=None,
                        )
                        break
                    except Exception as e:  # network, rate limit or unparseable output
                        rec["error"] = f"{type(e).__name__}: {e}"[:500]
                        await asyncio.sleep(2 ** attempt * 5)
            async with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()

        await tqdm.gather(*(one(j) for j in todo), desc=out_path.parent.name + "/" + out_path.stem)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--judge", required=True, help="<backend>:<model>, e.g. litellm:gpt-5.5-2026-04-23")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--first-rep", type=int, default=1, help="Index of the first repeat file (to add more later).")
    p.add_argument("--cases-per-stage", type=int, default=None, help="Seeded stratified case subset (default: all cases).")
    p.add_argument("--configs", nargs="*", default=None, help="Restrict to these pipeline::model configurations.")
    p.add_argument("--temperature", type=float, default=None, help="Default: provider default, as in the original grading.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    entries = load_entries()
    cases = sample_cases(entries, args.cases_per_stage, SEED)
    jobs = []
    for config, e in entries:
        if cases is not None and e.prompt_id not in cases:
            continue
        if args.configs and config not in args.configs:
            continue
        for idx, rubric in enumerate(e.rubrics):
            jobs.append({"prompt_id": e.prompt_id, "config": config, "criterion_idx": idx, "prompt": build_grader_prompt(e, rubric)})

    n_cases = len({j["prompt_id"] for j in jobs})
    n_configs = len({j["config"] for j in jobs})
    in_tokens = sum(len(j["prompt"]) for j in jobs) / CHARS_PER_TOKEN
    print(
        f"Judge {args.judge}: {n_cases} cases x {n_configs} configs = {len(jobs)} calls per repeat, "
        f"x{args.repeats} repeats = {len(jobs) * args.repeats} calls; "
        f"~{in_tokens * args.repeats / 1e6:.1f}M input tokens (+ ~150 output tokens per call, plus any reasoning tokens)."
    )
    if cases is not None:
        print("Cases:", ",".join(sorted(cases, key=int)))
    if args.dry_run:
        return

    judge = Judge(args.judge, args.temperature, args.seed)
    out_dir = ensure_output_dir() / "regrades" / judge_slug(args.judge)
    meta = {
        "judge": args.judge,
        "temperature": args.temperature,
        "seed": args.seed,
        "cases": sorted(cases, key=int) if cases is not None else "all",
        "configs": args.configs or "all",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for k in range(args.first_rep, args.first_rep + args.repeats):
        asyncio.run(run_rep(judge, jobs, out_dir / f"rep{k}.jsonl", args.concurrency))


if __name__ == "__main__":
    main()
