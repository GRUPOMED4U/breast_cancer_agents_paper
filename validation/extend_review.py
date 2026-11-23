"""Second workbook for a reviewer: every case in the dataset they have not seen yet.

The first workbooks covered a 36-case sample. This script brings one reviewer to
full coverage of the dataset: their part-2 workbook holds every case that was
not in their first workbook, so (first workbook + part 2) == all cases. The
script checks that before writing.

Cases not in the original allocation (never-sampled cases, including the stage 0
/ DCIS cases flagged in #12) form an "extension pool". The pool's case codes and
selected responses are created once, independently of the reviewer, and reused
for every reviewer. So when both oncologists review a case, they see the same
code and the same responses and their answers can be compared. Cases that were
already allocated (the other reviewer's unique cases, calibration cases) reuse
their existing code and responses for the same reason.

Note for the analysis: the calibration cases were discussed jointly during
training, so answers on CAL1/CAL2 are not independent between reviewers.

Inputs (from the first run): validation/output/allocation.csv,
selected_responses.jsonl, answer_key_responses.csv.
Outputs (gitignored), created once and then reused:
  validation/output/allocation_extension.csv
  validation/output/selected_responses_extension.jsonl
  validation/output/answer_key_responses_extension.csv
  validation/output/answer_key_rubrics_extension.csv
  validation/output/blinding_review_extension.tsv
Per reviewer:
  validation/output/workbooks/revisao_oncologista_<n>_parte2.xlsx
  validation/output/assignments_oncologista_<n>_parte2.csv

Usage:
    .venv/Scripts/python.exe validation/extend_review.py --reviewer 1
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import random

from build_workbooks import build_workbook
from common import SEED, STAGE0_PATTERN, ensure_output_dir, guard_overwrite, load_cases
from select_responses import assign_targets, load_candidates, select, write_outputs

EXT_SEED = SEED + 100


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_responses(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[(r["case_code"], r["slot"])] = r
    return out


def build_extension_pool(out_dir, cases, allocation) -> None:
    """Assign codes and select responses for every case not in the original allocation."""
    allocated = {a["prompt_id"] for a in allocation}
    pool = sorted((pid for pid in cases if pid not in allocated), key=int)
    rng = random.Random(EXT_SEED)

    first_new = len([a for a in allocation if a["case_code"].startswith("R")]) + 1
    codes = [f"R{i:02d}" for i in range(first_new, first_new + len(pool))]
    rng.shuffle(codes)
    ext_alloc = []
    for pid, code in zip(pool, codes):
        c = cases[pid]
        stage0 = bool(STAGE0_PATTERN.search(c["case_summary"]))
        rubrics = c["rubrics"]
        ext_alloc.append({
            "prompt_id": pid, "stage": str(c["breast_cancer_stage"]),
            "group": "extension_stage0" if stage0 else "extension", "case_code": code,
            "n_criteria": len(rubrics), "n_negative": sum(r["points"] < 0 for r in rubrics),
        })
    ext_alloc.sort(key=lambda a: a["case_code"])
    with open(out_dir / "allocation_extension.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ext_alloc[0].keys()))
        writer.writeheader()
        writer.writerows(ext_alloc)

    # Keep spreading responses over configurations: start from the first run's usage.
    config_uses, topup_uses = collections.Counter(), collections.Counter()
    for k in read_csv(out_dir / "answer_key_responses.csv"):
        (config_uses if k["slot"] == "primary" else topup_uses)[f"{k['pipeline']}::{k['model']}"] += 1

    by_case = load_candidates(rng)
    targets = assign_targets(ext_alloc, rng)
    selected, key_rows, config_uses = select(ext_alloc, cases, by_case, targets, rng, config_uses, topup_uses)
    write_outputs(out_dir, selected, key_rows, config_uses, suffix="_extension")

    responses = {(s["case_code"], s["slot"]): s for s in selected}
    with open(out_dir / "answer_key_rubrics_extension.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case_code", "prompt_id", "group", "rubric_idx", "points", "category", "judge_met_primary", "judge_met_topup", "criterion"])
        for a in ext_alloc:
            code, pid = a["case_code"], a["prompt_id"]
            for i, r in enumerate(cases[pid]["rubrics"]):
                writer.writerow([
                    code, pid, a["group"], i, r["points"], "|".join(r["tags"]),
                    responses[(code, "primary")]["judge_met"][i],
                    responses[(code, "topup")]["judge_met"][i] if r["points"] < 0 else "",
                    r["criterion"],
                ])
    print(f"Extension pool: {len(ext_alloc)} cases, codes {min(codes)}-{max(codes)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reviewer", type=int, choices=(1, 2), required=True)
    parser.add_argument("--force", action="store_true", help="Overwrite this reviewer's existing part-2 workbook.")
    args = parser.parse_args()

    out_dir = ensure_output_dir()
    cases = load_cases()
    allocation = read_csv(out_dir / "allocation.csv")

    if not (out_dir / "allocation_extension.csv").exists():
        build_extension_pool(out_dir, cases, allocation)
    else:
        print("Reusing the existing extension pool (same codes and responses for every reviewer).")
    ext_alloc = read_csv(out_dir / "allocation_extension.csv")

    all_alloc = allocation + ext_alloc
    alloc_by_code = {a["case_code"]: a for a in all_alloc}
    responses = read_responses(out_dir / "selected_responses.jsonl")
    responses.update(read_responses(out_dir / "selected_responses_extension.jsonl"))

    # Integrity: every dataset case has exactly one code and a primary + top-up response.
    pids = [a["prompt_id"] for a in all_alloc]
    assert len(pids) == len(set(pids)) == len(cases) and set(pids) == set(cases), "Case codes do not cover the dataset exactly once."
    assert all((a["case_code"], s) in responses for a in all_alloc for s in ("primary", "topup")), "Missing selected responses."

    seen = [a for a in allocation if a["group"] in ("shared", f"onc{args.reviewer}")]
    seen_codes = {a["case_code"] for a in seen}
    pending = [a for a in all_alloc if a["case_code"] not in seen_codes]

    # The two workbooks together must be the whole dataset, with no overlap.
    assert len(seen) + len(pending) == len(cases), f"{len(seen)} + {len(pending)} != {len(cases)}"
    assert {a["prompt_id"] for a in seen} | {a["prompt_id"] for a in pending} == set(cases)

    codes = [a["case_code"] for a in pending]
    random.Random(SEED + 10 + args.reviewer).shuffle(codes)
    wb_path = out_dir / "workbooks" / f"revisao_oncologista_{args.reviewer}_parte2.xlsx"
    guard_overwrite([wb_path], args.force)
    wb_path.parent.mkdir(exist_ok=True)
    build_workbook(wb_path, codes, cases, alloc_by_code, responses)

    source = {
        "onc1": "unique to oncologist 1 (part 1)", "onc2": "unique to oncologist 2 (part 1)",
        "calibration": "calibration case (discussed jointly in training)",
        "extension": "not in part-1 sample", "extension_stage0": "not in part-1 sample; stage 0 / DCIS (see #12)",
    }
    with open(out_dir / f"assignments_oncologista_{args.reviewer}_parte2.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["order", "case_code", "prompt_id", "stage", "source"])
        for i, code in enumerate(codes, start=1):
            a = alloc_by_code[code]
            writer.writerow([i, code, a["prompt_id"], a["stage"], source[a["group"]]])

    crit = sum(len(cases[alloc_by_code[c]["prompt_id"]]["rubrics"]) for c in codes)
    neg = sum(sum(r["points"] < 0 for r in cases[alloc_by_code[c]["prompt_id"]]["rubrics"]) for c in codes)
    by_source = collections.Counter(alloc_by_code[c]["group"] for c in codes)
    by_stage = collections.Counter(alloc_by_code[c]["stage"] for c in codes)
    print(f"Oncologist {args.reviewer}: part 1 = {len(seen)} cases, part 2 = {len(pending)} cases, total = {len(seen) + len(pending)} / {len(cases)} in the dataset")
    print(f"  part 2: {crit} criteria ({neg} negative), {len(codes)} primary + {len(codes)} top-up responses")
    print(f"  by source: {dict(by_source)}")
    print(f"  by stage: {dict(sorted(by_stage.items()))}")
    print(f"Wrote {wb_path}")


if __name__ == "__main__":
    main()
