"""Seeded, stage-stratified case allocation for the clinician review study (#13, #14).

Design (see #14): per stage, 3 shared cases (reviewed by both oncologists),
3 unique cases for oncologist 1 and 3 for oncologist 2, giving 36 distinct
cases and 48 case reviews. Two extra calibration cases are drawn from outside
the sample. Cases flagged as stage 0 / DCIS (#12) are excluded from the pool.

Output (gitignored): validation/output/allocation.csv

Usage:
    .venv/Scripts/python.exe validation/sample_cases.py [--per-group 3]
"""

from __future__ import annotations

import argparse
import collections
import csv
import random

from common import SEED, STAGE0_PATTERN, ensure_output_dir, load_cases

GROUPS = ("shared", "onc1", "onc2")
CALIBRATION_STAGES = ("2", "4")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-group", type=int, default=3, help="Cases per stage per group (3 = full design, 2 = reduced).")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cases = load_cases()

    excluded = sorted(
        (k for k, c in cases.items() if STAGE0_PATTERN.search(c["case_summary"])), key=int
    )
    by_stage: dict[str, list[str]] = collections.defaultdict(list)
    for k, c in sorted(cases.items(), key=lambda kv: int(kv[0])):
        if k not in excluded:
            by_stage[str(c["breast_cancer_stage"])].append(k)

    rows = []
    leftovers: dict[str, list[str]] = {}
    for stage in sorted(by_stage):
        pool = by_stage[stage][:]
        rng.shuffle(pool)
        need = args.per_group * len(GROUPS)
        if len(pool) < need:
            raise SystemExit(f"Stage {stage}: only {len(pool)} eligible cases, need {need}.")
        for g_idx, group in enumerate(GROUPS):
            for pid in pool[g_idx * args.per_group : (g_idx + 1) * args.per_group]:
                rows.append({"prompt_id": pid, "stage": stage, "group": group})
        leftovers[stage] = pool[need:]

    for stage in CALIBRATION_STAGES:
        rows.append({"prompt_id": leftovers[stage][0], "stage": stage, "group": "calibration"})

    # Random study codes so the code reveals neither stage nor group.
    review_rows = [r for r in rows if r["group"] != "calibration"]
    codes = [f"R{i:02d}" for i in range(1, len(review_rows) + 1)]
    rng.shuffle(codes)
    for r, code in zip(review_rows, codes):
        r["case_code"] = code
    for i, r in enumerate((r for r in rows if r["group"] == "calibration"), start=1):
        r["case_code"] = f"CAL{i}"

    # Rubric composition, for the balance check.
    for r in rows:
        rubrics = cases[r["prompt_id"]]["rubrics"]
        r["n_criteria"] = len(rubrics)
        r["n_negative"] = sum(x["points"] < 0 for x in rubrics)
        cats = collections.Counter(t for x in rubrics for t in x["tags"])
        for cat in ("systemic_therapy", "radiotherapy", "surgery", "complementary_exams"):
            r[f"n_{cat}"] = cats.get(cat, 0)

    out = ensure_output_dir() / "allocation.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["case_code"]))

    print(f"Excluded (stage 0 / DCIS, see #12): prompt_ids {excluded}")
    print(f"Wrote {out}")
    summary = collections.defaultdict(collections.Counter)
    for r in rows:
        s = summary[r["group"]]
        s["cases"] += 1
        for key in ("n_criteria", "n_negative", "n_systemic_therapy", "n_radiotherapy", "n_surgery", "n_complementary_exams"):
            s[key] += r[key]
    for group, s in summary.items():
        print(f"  {group:12s} " + ", ".join(f"{k}={v}" for k, v in s.items()))
    for onc in ("onc1", "onc2"):
        crit = summary["shared"]["n_criteria"] + summary[onc]["n_criteria"]
        print(f"  workload {onc}: {summary['shared']['cases'] + summary[onc]['cases']} cases, {crit} criteria")


if __name__ == "__main__":
    main()
