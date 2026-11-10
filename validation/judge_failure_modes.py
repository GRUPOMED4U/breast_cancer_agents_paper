"""Known judge failure modes, computed from the existing grades (#11, option F).

No API calls. Uses the per-criterion verdicts already stored in ``evals/``
(Gemini 2.5 Flash judge, one verdict per response x criterion).

1. Length: does the judge's met-rate rise with response length? Length is
   compared *within a case* (rank of the response among that case's
   configurations), so case difficulty and rubric size are held fixed.
   Reported separately for positive and negative criteria, since a longer
   response legitimately covers more positive criteria. A rising met-rate on
   *negative* criteria with length, or a rise on positive criteria that is
   much steeper than the score gain clinicians would give, is the signal to
   look for (R4-M12). Length is confounded with pipeline (D&C+SA responses are
   about 2x longer), so the within-configuration slope is also reported.
2. "Missing information" exception: how often a criterion is marked met with
   an explanation that invokes missing information in the case summary. These
   verdicts are exported for a spot-check, since the rule may be too lenient.
3. Parse failures: verdicts with an empty explanation, which the evaluator
   silently records as "not met".

Outputs:
  validation/results/judge_failure_modes.json      aggregate numbers only (safe to commit)
  validation/output/missing_info_spotcheck.csv     sampled verdicts with case text (gitignored)

Usage:
    .venv/Scripts/python.exe validation/judge_failure_modes.py
"""

from __future__ import annotations

import collections
import csv
import json
import random
import re
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from judge_common import RESULTS_DIR, SEED, ensure_output_dir, load_evaluated_responses, response_text

N_BOOT = 2000
N_SPOTCHECK = 60

# Explanations that rely on the "missing information" rule of GRADER_TEMPLATE.
MISSING_INFO = re.compile(
    r"\bmissing\b|not (?:provided|specified|stated|available|included|reported|mentioned in the (?:case|query|summary))"
    r"|\babsent\b|lack(?:s|ing)? (?:of )?(?:information|data)|information (?:is|was) not"
    r"|\bunknown\b|n[aã]o (?:informad|fornecid|dispon[ií]vel)|ausente",
    re.IGNORECASE,
)


def cluster_boot(values_by_case: dict[str, list[float]], rng: np.random.Generator) -> tuple[float, float, float]:
    """Mean of per-item values with a 95% CI from a bootstrap over cases."""
    keys = list(values_by_case)
    point = float(np.mean([v for k in keys for v in values_by_case[k]]))
    means = []
    for _ in range(N_BOOT):
        sample = rng.choice(len(keys), len(keys))
        vals = [v for i in sample for v in values_by_case[keys[i]]]
        means.append(np.mean(vals))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return point, float(lo), float(hi)


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = load_evaluated_responses()
    for r in rows:
        r["words"] = len(response_text(r["agent_response"]).split())

    by_case = collections.defaultdict(list)
    for r in rows:
        by_case[r["prompt_id"]].append(r)

    # Within-case length quintile (0 = shortest fifth of that case's responses).
    for case_rows in by_case.values():
        ranked = sorted(case_rows, key=lambda r: r["words"])
        for rank, r in enumerate(ranked):
            r["len_q"] = rank * 5 // len(ranked)

    # Long format: one record per (response, criterion).
    items = []
    for r in rows:
        for idx, (rub, g) in enumerate(zip(r["rubrics"], r["grader_responses"])):
            items.append(
                {
                    "prompt_id": r["prompt_id"],
                    "config": r["config"],
                    "criterion_idx": idx,
                    "sign": "positive" if rub["points"] > 0 else "negative",
                    "tags": rub.get("tags", []),
                    "met": bool(g.get("criteria_met")),
                    "explanation": g.get("explanation", ""),
                    "words": r["words"],
                    "len_q": r["len_q"],
                }
            )

    out: dict = {
        "n_responses": len(rows),
        "n_verdicts": len(items),
        "judge": sorted({r["eval_model"] for r in rows}),
        "overall_met_rate": float(np.mean([i["met"] for i in items])),
    }

    # --- 1. Length ---------------------------------------------------------
    quintiles = {}
    for sign in ("positive", "negative"):
        quintiles[sign] = {}
        for q in range(5):
            sel = [i for i in items if i["sign"] == sign and i["len_q"] == q]
            quintiles[sign][f"Q{q + 1}"] = {
                "n": len(sel),
                "met_rate": float(np.mean([i["met"] for i in sel])),
                "median_words": float(np.median([i["words"] for i in sel])),
            }
    out["met_rate_by_within_case_length_quintile"] = quintiles

    # Within-criterion Spearman rho between length and met, across the
    # configurations that answered that case; averaged over criteria.
    def criterion_rhos(group_key) -> dict[str, dict[str, list[float]]]:
        groups = collections.defaultdict(list)
        for i in items:
            groups[group_key(i)].append(i)
        res = {"positive": collections.defaultdict(list), "negative": collections.defaultdict(list)}
        for grp in groups.values():
            met = [i["met"] for i in grp]
            if len(set(met)) < 2:
                continue  # constant verdicts: no information about length
            rho = spearmanr([i["words"] for i in grp], met).statistic
            if np.isfinite(rho):
                res[grp[0]["sign"]][grp[0]["prompt_id"]].append(float(rho))
        return res

    rho_all = criterion_rhos(lambda i: (i["prompt_id"], i["criterion_idx"]))
    out["within_criterion_spearman_length_vs_met"] = {
        sign: dict(zip(("mean_rho", "ci_low", "ci_high"), cluster_boot(v, rng)))
        | {"n_criteria_with_variation": sum(len(x) for x in v.values())}
        for sign, v in rho_all.items()
    }

    # Same, but within pipeline: does a longer response from the *same*
    # pipeline get more criteria met? Removes the pipeline-length confound.
    rho_pipe = criterion_rhos(lambda i: (i["prompt_id"], i["criterion_idx"], i["config"].split("::")[0]))
    out["within_criterion_within_pipeline_spearman"] = {
        sign: dict(zip(("mean_rho", "ci_low", "ci_high"), cluster_boot(v, rng)))
        | {"n_groups_with_variation": sum(len(x) for x in v.values())}
        for sign, v in rho_pipe.items()
    }

    # Case score vs length, within case.
    score_rhos = collections.defaultdict(list)
    for pid, case_rows in by_case.items():
        rho = spearmanr([r["words"] for r in case_rows], [r["score"] for r in case_rows]).statistic
        if np.isfinite(rho):
            score_rhos[pid].append(float(rho))
    out["within_case_spearman_length_vs_case_score"] = dict(
        zip(("mean_rho", "ci_low", "ci_high"), cluster_boot(score_rhos, rng))
    )

    # --- 2. Missing-information exception ----------------------------------
    flagged = [i for i in items if i["met"] and MISSING_INFO.search(i["explanation"])]
    mi = {}
    for sign in ("positive", "negative"):
        met_sign = [i for i in items if i["met"] and i["sign"] == sign]
        fl = [i for i in flagged if i["sign"] == sign]
        mi[sign] = {"met_verdicts": len(met_sign), "flagged": len(fl), "share": len(fl) / max(1, len(met_sign))}
    mi["by_category"] = {}
    tag_counts = collections.Counter(t for i in items if i["met"] for t in i["tags"])
    tag_flag = collections.Counter(t for i in flagged for t in i["tags"])
    for t in sorted(tag_counts):
        mi["by_category"][t] = {"met_verdicts": tag_counts[t], "flagged": tag_flag[t], "share": tag_flag[t] / tag_counts[t]}
    mi["note"] = "Keyword flag over the judge's explanation; an upper bound on use of the exception. See the spot-check file."
    out["missing_info_exception"] = mi

    # --- 3. Parse failures --------------------------------------------------
    out["empty_explanations"] = sum(1 for i in items if not i["explanation"])

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "judge_failure_modes.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Spot-check export (patient data -> gitignored output dir).
    cases_rows = {(r["prompt_id"], r["config"]): r for r in rows}
    pyrng = random.Random(SEED)
    sample = pyrng.sample(flagged, min(N_SPOTCHECK, len(flagged)))
    path = ensure_output_dir() / "missing_info_spotcheck.csv"
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prompt_id", "config", "criterion_idx", "sign", "points", "criterion", "judge_explanation", "exception_justified (Y/N/unclear)", "notes"])
        for i in sorted(sample, key=lambda i: (int(i["prompt_id"]), i["config"], i["criterion_idx"])):
            rub = cases_rows[(i["prompt_id"], i["config"])]["rubrics"][i["criterion_idx"]]
            w.writerow([i["prompt_id"], i["config"], i["criterion_idx"], i["sign"], rub["points"], rub["criterion"], i["explanation"], "", ""])

    print(json.dumps(out, indent=2))
    print(f"\nSpot-check sample: {path}")


if __name__ == "__main__":
    main()
