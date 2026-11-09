"""Descriptive rubric analyses for the AIRG transparency work (option F in #10, task 4 in #13).

Produces aggregate tables (by stage, category and sign), a point-value
distribution figure, example rubric sets for the Supplement (with dates masked),
and a consistency check between results/ and evals/.

Outputs (gitignored; copy reviewed aggregates into the paper by hand):
  validation/output/descriptives/*.csv, *.tex, *.png, report.md

Usage:
    .venv/Scripts/python.exe validation/rubric_descriptives.py
"""

from __future__ import annotations

import collections
import csv
import json
import random
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from common import EVALS_DIR, REPO_ROOT, SEED, STAGE0_PATTERN, ensure_output_dir, load_cases  # noqa: E402

CATEGORY_LABEL = {
    "systemic_therapy": "Systemic therapy",
    "radiotherapy": "Radiotherapy",
    "surgery": "Surgery",
    "complementary_exams": "Complementary exams",
}
DATE_RE = re.compile(r"\b(?:\d{1,2}/)?\d{1,2}/\d{4}\b|\b\d{1,2}/\d{2}\b(?=\D)|\b(?:19|20)\d{2}\b")


def iqr(s: pd.Series) -> str:
    return f"{s.median():g} [{s.quantile(0.25):g}–{s.quantile(0.75):g}]"


def latex_escape(text: str) -> str:
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        text = text.replace(a, b)
    return text


def main() -> None:
    out = ensure_output_dir() / "descriptives"
    out.mkdir(exist_ok=True)
    cases = load_cases()

    rows = []
    for pid, c in cases.items():
        for i, r in enumerate(c["rubrics"]):
            for tag in r["tags"]:
                rows.append({"prompt_id": pid, "stage": str(c["breast_cancer_stage"]), "rubric_idx": i,
                             "category": tag, "points": r["points"], "sign": "positive" if r["points"] > 0 else "negative",
                             "n_tags": len(r["tags"])})
    df = pd.DataFrame(rows)
    crit = df.drop_duplicates(["prompt_id", "rubric_idx"])  # one row per criterion
    report = ["# AIRG rubric descriptives", ""]
    report.append(f"- Cases: {crit.prompt_id.nunique()}; criteria: {len(crit)}; criteria with >1 category tag: {(crit.n_tags > 1).sum()}")
    report.append(f"- Positive criteria: {(crit.points > 0).sum()} ({(crit.points > 0).mean():.1%}); negative: {(crit.points < 0).sum()} ({(crit.points < 0).mean():.1%}); zero-point: {(crit.points == 0).sum()}")
    report.append("")

    # 1. Criteria per case by stage.
    per_case = crit.groupby(["prompt_id", "stage"]).agg(
        n=("points", "size"), n_pos=("points", lambda p: (p > 0).sum()), n_neg=("points", lambda p: (p < 0).sum()),
        pos_points=("points", lambda p: p[p > 0].sum()), neg_points=("points", lambda p: p[p < 0].sum()),
    ).reset_index()
    t1 = per_case.groupby("stage").agg(
        cases=("n", "size"), criteria=("n", "sum"), per_case=("n", iqr), positive_per_case=("n_pos", iqr),
        negative_per_case=("n_neg", iqr), positive_points_per_case=("pos_points", iqr), negative_points_per_case=("neg_points", iqr),
    )
    t1.loc["all"] = [len(per_case), per_case.n.sum(), iqr(per_case.n), iqr(per_case.n_pos), iqr(per_case.n_neg), iqr(per_case.pos_points), iqr(per_case.neg_points)]
    t1.to_csv(out / "criteria_per_case_by_stage.csv")
    report += ["## Criteria per case by stage (median [IQR])", "", t1.to_markdown(), ""]

    # 2. Category x sign (counts by tag; a multi-tag criterion counts in each category).
    t2 = df.groupby(["category", "sign"]).points.agg(["size", "mean", "median", "min", "max"]).round(2).unstack("sign")
    t2.index = [CATEGORY_LABEL.get(i, i) for i in t2.index]
    t2.to_csv(out / "weights_by_category_and_sign.csv")
    report += ["## Point values by category and sign (counted per tag)", "", t2.to_markdown(), ""]

    # 3. Stage x category share of criteria.
    t3 = pd.crosstab(df.stage, df.category.map(CATEGORY_LABEL), normalize="index").round(3)
    t3.to_csv(out / "category_share_by_stage.csv")
    report += ["## Share of criteria by category within each stage", "", t3.to_markdown(), ""]

    # 4. Point-value distribution figure.
    values = list(range(-10, 11))
    fig, ax = plt.subplots(figsize=(8, 4))
    bottom = pd.Series(0, index=values)
    for cat, label in CATEGORY_LABEL.items():
        counts = df[df.category == cat].points.value_counts().reindex(values, fill_value=0)
        ax.bar(values, counts, bottom=bottom, label=label)
        bottom += counts
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("Rubric point value")
    ax.set_ylabel("Number of criteria (per category tag)")
    ax.set_xticks(values)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "point_value_distribution.png", dpi=200)
    pv = crit.points.value_counts().reindex(values, fill_value=0)
    report += ["## Point value counts (per criterion)", "", pv[pv > 0].to_frame("criteria").T.to_markdown(), ""]

    # 5. Example rubric sets: one case per stage, outside the review sample if the allocation exists.
    alloc_path = REPO_ROOT / "validation" / "output" / "allocation.csv"
    in_review = set()
    if alloc_path.exists():
        with open(alloc_path, encoding="utf-8") as f:
            in_review = {r["prompt_id"] for r in csv.DictReader(f)}
    rng = random.Random(SEED)
    tex = []
    for stage in sorted({str(c["breast_cancer_stage"]) for c in cases.values()}):
        pool = sorted(k for k, c in cases.items() if str(c["breast_cancer_stage"]) == stage and k not in in_review and not STAGE0_PATTERN.search(c["case_summary"]))
        pid = rng.choice(pool)
        rubrics = sorted(cases[pid]["rubrics"], key=lambda r: (-r["points"]))
        tex.append(f"% Example case, stage {stage} (internal prompt_id {pid}). Dates masked as [date]; review before publishing.")
        tex.append(r"\begin{longtable}{p{0.62\textwidth} r p{0.2\textwidth}}")
        tex.append(rf"\caption{{Example AIRG rubric set for a stage {stage} case.}}\\")
        tex.append(r"\hline \textbf{Criterion} & \textbf{Points} & \textbf{Category} \\ \hline \endhead")
        for r in rubrics:
            crit_text = latex_escape(DATE_RE.sub("[date]", r["criterion"]))
            cats = ", ".join(CATEGORY_LABEL.get(t, t) for t in r["tags"])
            tex.append(f"{crit_text} & {r['points']:+d} & {cats} \\\\")
        tex.append(r"\hline \end{longtable}")
        tex.append("")
    (out / "example_rubrics.tex").write_text("\n".join(tex), encoding="utf-8")
    report += ["## Example rubric sets", "", "Written to `example_rubrics.tex` (one case per stage, outside the review sample, dates masked). Check every criterion for identifying details before adding it to the Supplement.", ""]

    # 6. Consistency check: results/ vs evals/.
    report += ["## results/ vs evals/ consistency", "", "| pipeline | model | results rows | unique cases | duplicate rows | evaluated (evals/) |", "|---|---|---|---|---|---|"]
    evaluated = {(p.parent.name, p.stem) for p in EVALS_DIR.glob("*/*.json")}
    for path in sorted((REPO_ROOT / "results").glob("*/*.jsonl")):
        ids = [str(json.loads(l)["prompt_id"]) for l in open(path, encoding="utf-8")]
        dup = len(ids) - len(set(ids))
        report.append(f"| {path.parent.name} | {path.stem} | {len(ids)} | {len(set(ids))} | {dup} | {'yes' if (path.parent.name, path.stem) in evaluated else 'no'} |")
    report.append("")

    (out / "report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\nWrote outputs to {out}")


if __name__ == "__main__":
    main()
