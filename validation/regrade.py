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

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_models import EvaluationEntry, GraderResponse  # noqa: E402
from evaluator import build_grader_prompt  # noqa: E402

from judge_common import EVALS_DIR, OUTPUT_DIR, SEED, ensure_output_dir, maybe_literal  # noqa: E402


CHARS_PER_TOKEN = 4.0  # calibrated on the original grading (78.0M prompt tokens for 30,969 calls)
MAX_ATTEMPTS = 6
# Errors that mean "no more credit / bad key": stop the run instead of retrying.
FATAL_PATTERNS = re.compile(
    r"insufficient_quota|exceeded your current quota|insufficient credits|credit balance|"
    r"payment required|\b402\b|billing|invalid api key|incorrect api key|AuthenticationError",
    re.IGNORECASE,
)


class CreditsExhausted(Exception):
    pass
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
    def __init__(
        self,
        spec: str,
        temperature: float | None,
        seed: int | None,
        reasoning_effort: str | None = None,
        openrouter_provider: str | None = None,
    ):
        self.backend, self.model = spec.split(":", 1)
        self.temperature, self.seed = temperature, seed
        self.reasoning_effort = reasoning_effort
        self.openrouter_provider = openrouter_provider
        if self.backend == "gemini":
            from google import genai

            self.client = genai.Client()
        elif self.backend == "litellm":
            import litellm

            litellm.suppress_debug_info = True
            self.litellm = litellm
        else:
            raise SystemExit(f"Unknown backend '{self.backend}' (use gemini: or litellm:)")

    async def grade(self, prompt: str, cache_key: str | None = None) -> dict:
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
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if cache_key is not None and "gpt" in self.model:
            # Routes requests that share a prefix to the same cache (OpenAI).
            kwargs["prompt_cache_key"] = cache_key
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if self.openrouter_provider:
            # Serve only from this upstream provider, never fall back to another host.
            kwargs["extra_body"] = {"provider": {"order": [self.openrouter_provider], "allow_fallbacks": False}}
        r = await self.litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            drop_params=True,
            **kwargs,
        )
        usage = r.usage.model_dump() if getattr(r, "usage", None) else {}
        cost = (getattr(r, "_hidden_params", None) or {}).get("response_cost")
        return {
            "text": r.choices[0].message.content,
            "model_version": r.model,
            "served_by": getattr(r, "provider", None),
            "usage": usage,
            "cost": cost,
        }


async def run_rep(judge: Judge, jobs: list[dict], out_path: Path, concurrency: int) -> bool:
    """Grade the jobs not yet in ``out_path``. Returns False if the run was
    stopped because the provider reported exhausted credits or a bad key."""
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
        return True

    # Group by response. Within a response, grade one criterion first and the
    # rest after it, so the shared prefix (case summary + response) is cached
    # by providers with automatic prompt caching.
    by_response = collections.defaultdict(list)
    for j in todo:
        by_response[(j["prompt_id"], j["config"])].append(j)

    call_sem = asyncio.Semaphore(concurrency)
    response_sem = asyncio.Semaphore(max(2, concurrency // 4))
    stop = asyncio.Event()
    lock = asyncio.Lock()
    stats = collections.Counter()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bar = tqdm(total=len(todo), desc=out_path.parent.name + "/" + out_path.stem)

    with open(out_path, "a", encoding="utf-8") as f:

        async def one(job):
            if stop.is_set():
                return
            rec = {k: job[k] for k in ("prompt_id", "config", "criterion_idx")}
            rec.update(judge=f"{judge.backend}:{judge.model}", criteria_met=None, explanation="",
                       model_version=None, usage={}, cost=None, error=None)
            async with call_sem:
                for attempt in range(MAX_ATTEMPTS):
                    if stop.is_set():
                        return
                    try:
                        raw = await judge.grade(job["prompt"], cache_key=f"{job['prompt_id']}|{job['config']}")
                        verdict = parse_verdict(raw["text"])
                        rec.update(
                            criteria_met=verdict.criteria_met,
                            explanation=verdict.explanation,
                            model_version=raw["model_version"],
                            usage=raw["usage"],
                            cost=raw.get("cost"),
                            served_by=raw.get("served_by"),
                            error=None,
                        )
                        break
                    except Exception as e:  # network, rate limit or unparseable output
                        msg = f"{type(e).__name__}: {e}"
                        rec["error"] = msg[:500]
                        if FATAL_PATTERNS.search(msg):
                            if not stop.is_set():
                                print(f"\nStopping: provider reports exhausted credits or a bad key: {msg[:300]}")
                            stop.set()
                            return
                        await asyncio.sleep(min(60, 5 * 2**attempt))
            async with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                stats["written"] += 1
                stats["failed"] += rec["criteria_met"] is None
                stats["cost"] += rec["cost"] or 0
                bar.update(1)

        async def response(js):
            async with response_sem:
                if stop.is_set():
                    return
                await one(js[0])
                await asyncio.gather(*(one(j) for j in js[1:]))

        await asyncio.gather(*(response(js) for js in by_response.values()))
    bar.close()
    print(f"Written {stats['written']} verdicts ({stats['failed']} failed), reported cost ${stats['cost']:.2f}")
    return not stop.is_set()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--judge", required=True, help="<backend>:<model>, e.g. litellm:gpt-5.5-2026-04-23")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--first-rep", type=int, default=1, help="Index of the first repeat file (to add more later).")
    p.add_argument("--cases-per-stage", type=int, default=None, help="Seeded stratified case subset (default: all cases).")
    p.add_argument("--configs", nargs="*", default=None, help="Restrict to these pipeline::model configurations.")
    p.add_argument("--temperature", type=float, default=None, help="Default: provider default, as in the original grading.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--openrouter-provider", default=None, help="OpenRouter only: pin the upstream provider (e.g. openai), no fallbacks.")
    p.add_argument("--reasoning-effort", default=None, help="e.g. low / medium / high (LiteLLM judges). Default: provider default.")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument(
        "--slug",
        default=None,
        help="Output folder name (default: from the model id). Use the same slug to continue a "
        "judge through another provider, e.g. OpenRouter.",
    )
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

    judge = Judge(args.judge, args.temperature, args.seed, args.reasoning_effort, args.openrouter_provider)
    out_dir = ensure_output_dir() / "regrades" / (args.slug or judge_slug(args.judge))
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    run = {
        "judge": args.judge,
        "temperature": args.temperature,
        "seed": args.seed,
        "reasoning_effort": args.reasoning_effort,
        "openrouter_provider": args.openrouter_provider,
        "cases": sorted(cases, key=int) if cases is not None else "all",
        "configs": args.configs or "all",
    }
    # Top-level keys describe the first run; later runs (e.g. another provider) are listed in "runs".
    for k, v in run.items():
        meta.setdefault(k, v)
    meta.setdefault("runs", []).append(run)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for k in range(args.first_rep, args.first_rep + args.repeats):
        if not asyncio.run(run_rep(judge, jobs, out_dir / f"rep{k}.jsonl", args.concurrency)):
            raise SystemExit(2)


if __name__ == "__main__":
    main()
