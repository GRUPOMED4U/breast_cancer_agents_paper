"""Choose the blinded responses that oncologists will review (#13, #14).

For each allocated case:
  * primary response: rated holistically, and every criterion is marked
    met / not met. Across the sample the primary responses are balanced over
    within-case score terciles (low / mid / high among that case's 27
    configurations) and spread over configurations (each used at most ~2 times).
  * top-up response (optional step 5): a different configuration, used only
    for the case's negative criteria. It is enriched for responses where the
    judge marked at least one negative criterion as met, so the judge's
    sensitivity on negatives can be estimated. Because of this enrichment,
    top-up items must be analysed separately from the primary items.

Responses are blinded: the EHR patient ID is replaced with the case code, and
phrases that reveal the pipeline (specialist agents, fact checking) are
replaced with "[removido]". Responses that are refusals or very short are not
eligible.

Inputs: validation/output/allocation.csv (from sample_cases.py) and evals/.
Outputs (gitignored):
  validation/output/selected_responses.jsonl  blinded texts + key, for build_workbooks.py
  validation/output/answer_key_responses.csv  which configuration is behind each response

Usage:
    .venv/Scripts/python.exe validation/select_responses.py
"""

from __future__ import annotations

import collections
import csv
import json
import random
import re

from common import SEED, ensure_output_dir, load_cases, load_evaluated_responses, response_text

MIN_WORDS = 150
REFUSAL = re.compile(r"\bas an ai\b|unable to perform the requested task|não posso (ajudar|realizar)", re.IGNORECASE)

# Phrases that reveal the pipeline. Deliberately specific: "agents" alone is
# common in pharmacology ("bone-modifying agents") and must be kept.
GIVEAWAYS = [
    r"[^.\n]*\b(?:fact[- ]?check\w*|fact checker)\b[^.\n]*[.:]?",
    r"[^.\n]*\bverificador(?:es)? de fatos\b[^.\n]*[.:]?",
    r"\b(?:e |and )?(?:nas |the )?(?:análises|análise|consultations?|analyses|input|inputs|opiniões|pareceres)? ?(?:d[oa]s )?(?:dos )?agentes? especialistas?\b",
    r"\b(?:e |and )?(?:the )?(?:expert|specialist|specialized|especialista)s? agents?(?: consultations?| analyses| input| reports?)?\b",
    r"\bagentes? (?:especializad[oa]s?|de (?:terapia|radioterapia|cirurgia|exames)\w*)\b",
    r"\b(?:sub-?agent\w*|subagente\w*|orchestrator|orquestrador\w*)\b",
]
GIVEAWAY_RE = re.compile("|".join(f"(?:{g})" for g in GIVEAWAYS), re.IGNORECASE)

# Meta-commentary from revision/fact-checking stages. Not redacted automatically
# (removing it can leave odd text); responses containing it are avoided when
# possible and any that remain are listed for manual review.
META_RE = re.compile(
    r"revisad[oa]s? e verificad[oa]s?|verificad[oa] e revisad[oa]|required refinement|"
    r"corrected (?:and \w+ )?(?:clinical )?report|revised (?:version|report)|vers[aã]o revisada|"
    r"corre[cç][oõ]es (?:realizadas|feitas)|corrections? (?:made|applied)|\bdraft\b|rascunho|"
    r"(?:this|the) report (?:has been|was) (?:reviewed|verified)",
    re.IGNORECASE,
)
PATIENT_ID_RE = re.compile(r"ID da paciente:\s*(\d+)", re.IGNORECASE)
# Numbers introduced as a patient/case identifier, even when the model
# mis-copied the EHR ID (e.g., an extra digit).
ID_CONTEXT_RE = re.compile(
    r"(?P<prefix>\b(?:ID|caso|case|paciente|patient)\b[\s:#nº°.-]*(?:ID\b[\s:#.-]*)?(?:da paciente\s*)?)(?P<num>\d{5,8})\b",
    re.IGNORECASE,
)


def blind(text: str, patient_id: str | None, case_code: str) -> tuple[str, int, int]:
    n_ids = 0
    if patient_id:
        text, n_ids = re.subn(rf"\b{re.escape(patient_id)}\b", case_code, text)
    text, n_ctx = ID_CONTEXT_RE.subn(lambda m: m.group("prefix") + case_code, text)
    text, n_give = GIVEAWAY_RE.subn("[removido]", text)
    # Revision/fact-check stages sometimes prepend a preamble ("Several key areas
    # required refinement... Here is the revised report:") before the report's
    # first heading. Drop that preamble when it carries meta-commentary.
    heading = re.search(r"^#{1,4} ", text, re.MULTILINE)
    if heading and heading.start() > 0:
        preamble = text[: heading.start()]
        if ("[removido]" in preamble or META_RE.search(preamble)) and len(text) - heading.start() > 1000:
            text = "[removido]\n\n" + text[heading.start():]
            n_give += 1
    return text, n_ids + n_ctx, n_give


def main() -> None:
    rng = random.Random(SEED)
    out_dir = ensure_output_dir()
    alloc_path = out_dir / "allocation.csv"
    if not alloc_path.exists():
        raise SystemExit("Run validation/sample_cases.py first.")
    with open(alloc_path, encoding="utf-8") as f:
        allocation = list(csv.DictReader(f))

    cases = load_cases()
    by_case: dict[str, list[dict]] = collections.defaultdict(list)
    for r in load_evaluated_responses():
        text = response_text(r["agent_response"])
        r["text"] = text
        r["words"] = len(text.split())
        r["eligible"] = r["words"] >= MIN_WORDS and not REFUSAL.search(text)
        r["n_giveaways"] = len(GIVEAWAY_RE.findall(text)) + len(META_RE.findall(text))
        neg_idx = [i for i, x in enumerate(r["rubrics"]) if x["points"] < 0]
        r["neg_met"] = sum(bool(r["grader_responses"][i].get("criteria_met")) for i in neg_idx)
        by_case[r["prompt_id"]].append(r)

    # Within-case terciles over all evaluated configurations.
    for rows in by_case.values():
        ranked = sorted(rows, key=lambda r: (r["score"], rng.random()))
        n = len(ranked)
        for rank, r in enumerate(ranked):
            r["tercile"] = ("low", "mid", "high")[min(2, rank * 3 // n)]

    # Balanced target terciles: within each group and stage, one case per tercile
    # (when 3 cases per stage); otherwise rotate.
    targets: dict[str, str] = {}
    cells = collections.defaultdict(list)
    for a in allocation:
        cells[(a["group"], a["stage"])].append(a["case_code"])
    for key in sorted(cells):
        codes = sorted(cells[key])
        rng.shuffle(codes)
        terciles = ["low", "mid", "high"]
        rng.shuffle(terciles)
        for i, code in enumerate(codes):
            targets[code] = terciles[i % 3] if key[0] != "calibration" else "mid"

    config_uses = collections.Counter()
    topup_uses = collections.Counter()
    selected, key_rows = [], []
    for a in sorted(allocation, key=lambda a: a["case_code"]):
        code, pid = a["case_code"], a["prompt_id"]
        m = PATIENT_ID_RE.search(cases[pid]["case_summary"])
        patient_id = m.group(1) if m else None
        candidates = [r for r in by_case[pid] if r["eligible"]]

        pool = [r for r in candidates if r["tercile"] == targets[code]] or candidates
        primary = min(pool, key=lambda r: (config_uses[r["config"]], r["n_giveaways"] > 0, rng.random()))
        config_uses[primary["config"]] += 1

        rest = [r for r in candidates if r["config"] != primary["config"]]
        topup = min(
            rest,
            key=lambda r: (-min(r["neg_met"], 1), r["n_giveaways"] > 0, topup_uses[r["config"]], rng.random()),
        )
        topup_uses[topup["config"]] += 1

        for slot, r in (("primary", primary), ("topup", topup)):
            text, n_ids, n_give = blind(r["text"], patient_id, code)
            selected.append(
                {
                    "case_code": code,
                    "prompt_id": pid,
                    "slot": slot,
                    "text": text,
                    "judge_met": [bool(g.get("criteria_met")) for g in r["grader_responses"]],
                }
            )
            key_rows.append(
                {
                    "case_code": code,
                    "prompt_id": pid,
                    "stage": a["stage"],
                    "group": a["group"],
                    "slot": slot,
                    "pipeline": r["pipeline"],
                    "model": r["model"],
                    "case_score": round(r["score"], 4),
                    "within_case_tercile": r["tercile"],
                    "target_tercile": targets[code] if slot == "primary" else "",
                    "words": r["words"],
                    "judge_negatives_met": r["neg_met"],
                    "patient_ids_redacted": n_ids,
                    "pipeline_phrases_redacted": n_give,
                }
            )

    with open(out_dir / "selected_responses.jsonl", "w", encoding="utf-8") as f:
        for s in selected:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Everything a human should eyeball before the workbooks go out.
    review_lines = []
    check_re = re.compile(r"\[removido\]|\b\d{5,8}\b|" + META_RE.pattern, re.IGNORECASE)
    for s in selected:
        for m in check_re.finditer(s["text"]):
            ctx = s["text"][max(0, m.start() - 120) : m.end() + 100].replace("\n", " ")
            review_lines.append(f"{s['case_code']}\t{s['slot']}\t{m.group(0)}\t{ctx}")
    (out_dir / "blinding_review.tsv").write_text(
        "case_code\tslot\tmatch\tcontext\n" + "\n".join(review_lines) + "\n", encoding="utf-8"
    )
    with open(out_dir / "answer_key_responses.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(key_rows[0].keys()))
        writer.writeheader()
        writer.writerows(key_rows)

    prim = [k for k in key_rows if k["slot"] == "primary"]
    top = [k for k in key_rows if k["slot"] == "topup"]
    print(f"Selected {len(prim)} primary and {len(top)} top-up responses.")
    print("  primary terciles:", dict(collections.Counter(k["within_case_tercile"] for k in prim)))
    print("  primary pipelines:", dict(collections.Counter(k["pipeline"] for k in prim)))
    print("  primary models:", dict(collections.Counter(k["model"] for k in prim)))
    print("  max uses of one config (primary):", max(config_uses.values()), f"over {len(config_uses)} configs")
    print("  top-ups with >=1 negative criterion met by judge:", sum(k["judge_negatives_met"] > 0 for k in top), "/", len(top))
    print("  responses with patient IDs redacted:", sum(k["patient_ids_redacted"] > 0 for k in key_rows))
    print("  responses with pipeline phrases redacted:", sum(k["pipeline_phrases_redacted"] > 0 for k in key_rows))
    print(f"Wrote {out_dir / 'selected_responses.jsonl'} and {out_dir / 'answer_key_responses.csv'}")


if __name__ == "__main__":
    main()
