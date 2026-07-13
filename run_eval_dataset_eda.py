#!/usr/bin/env python3
"""
Run exploratory data analysis for evaluation datasets stored as JSONL.

The expected input is a JSONL file where each line is a JSON object. The script
is robust to missing columns, but it is optimized for datasets with fields such
as:

- prompt_id
- prompt
- case_summary
- official_recommendations
- breast_cancer_stage
- filename
- rubrics

Example:
    python eval_dataset_eda_cli.py \
        --input eval_dataset.jsonl \
        --output outputs/eval_dataset_eda \
        --zip
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DERIVED_COLUMNS = {
    "prompt_content",
    "case_summary_chars",
    "official_recommendations_chars",
    "case_summary_lines",
    "official_recommendations_lines",
    "n_prompt_messages",
    "prompt_roles",
    "n_rubrics",
    "rubric_total_points",
    "stage_numeric",
    "filename_stage",
    "filename_stage_matches",
    "prompt_equals_case_summary",
}


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load a JSONL file and collect parse errors without stopping execution."""
    records: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except Exception as exc:
                parse_errors.append(
                    {
                        "line": line_no,
                        "error": str(exc),
                        "preview": line[:300],
                    }
                )
                continue

            if isinstance(item, dict):
                records.append(item)
            else:
                parse_errors.append(
                    {
                        "line": line_no,
                        "error": f"Expected JSON object, got {type(item).__name__}",
                        "preview": str(item)[:300],
                    }
                )

    return records, parse_errors


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Ensure expected columns exist, filling missing ones with None."""
    for column in columns:
        if column not in df.columns:
            df[column] = None

    return df


def str_len(value: Any) -> float:
    """Return string length or NaN for non-string values."""
    return len(value) if isinstance(value, str) else np.nan


def line_count(value: Any) -> float:
    """Return line count or NaN for non-string values."""
    return value.count("\n") + 1 if isinstance(value, str) else np.nan


def get_prompt_content(prompt: Any) -> str | None:
    """Extract the content of the first prompt message."""
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
        return prompt[0].get("content")

    return None


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add reusable derived columns for EDA."""
    df = df.copy()

    df["prompt_content"] = df["prompt"].map(get_prompt_content)
    df["case_summary_chars"] = df["case_summary"].map(str_len)
    df["official_recommendations_chars"] = df["official_recommendations"].map(str_len)
    df["case_summary_lines"] = df["case_summary"].map(line_count)
    df["official_recommendations_lines"] = df["official_recommendations"].map(
        line_count
    )

    df["n_prompt_messages"] = df["prompt"].map(
        lambda prompt: len(prompt) if isinstance(prompt, list) else np.nan
    )
    df["prompt_roles"] = df["prompt"].map(
        lambda prompt: (
            ",".join([str(message.get("role", "")) for message in prompt])
            if isinstance(prompt, list)
            else None
        )
    )

    df["n_rubrics"] = df["rubrics"].map(
        lambda rubrics: len(rubrics) if isinstance(rubrics, list) else np.nan
    )
    df["rubric_total_points"] = df["rubrics"].map(
        lambda rubrics: (
            sum((item.get("points", 0) or 0) for item in rubrics)
            if isinstance(rubrics, list)
            else np.nan
        )
    )

    df["stage_numeric"] = pd.to_numeric(df["breast_cancer_stage"], errors="coerce")
    df["filename_stage"] = df["filename"].astype(str).str.extract(r"stage_(\d+)")
    df["filename_stage_matches"] = (
        df["breast_cancer_stage"].astype(str).eq(df["filename_stage"].astype(str))
    )
    df["prompt_equals_case_summary"] = df["prompt_content"].eq(
        "# Case summary\n" + df["case_summary"].astype(str)
    )

    return df


def flatten_rubrics(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten nested rubrics into one row per rubric item."""
    rows: list[dict[str, Any]] = []

    for record_index, record in enumerate(records):
        rubrics = record.get("rubrics") or []

        if not isinstance(rubrics, list):
            continue

        for rubric_index, rubric in enumerate(rubrics):
            if not isinstance(rubric, dict):
                continue

            tags = rubric.get("tags") or []
            if not isinstance(tags, list):
                tags = [tags]

            rows.append(
                {
                    "record_index": record_index,
                    "prompt_id": record.get("prompt_id"),
                    "breast_cancer_stage": record.get("breast_cancer_stage"),
                    "filename": record.get("filename"),
                    "rubric_index": rubric_index,
                    "criterion": rubric.get("criterion"),
                    "points": rubric.get("points"),
                    "tags": tags,
                    "n_tags": len(tags),
                    "criterion_chars": len(rubric.get("criterion") or ""),
                }
            )

    return pd.DataFrame(rows)


def flatten_tags(rubrics_df: pd.DataFrame) -> pd.DataFrame:
    """Explode rubric tags into one row per tag assignment."""
    if rubrics_df.empty:
        return pd.DataFrame(
            columns=[
                "record_index",
                "prompt_id",
                "breast_cancer_stage",
                "filename",
                "rubric_index",
                "criterion",
                "points",
                "tag",
                "n_tags",
                "criterion_chars",
            ]
        )

    return rubrics_df.explode("tags").rename(columns={"tags": "tag"})


def parse_case_summary_fields(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Parse top-level `Field: value` lines from case summaries."""
    rows: list[dict[str, Any]] = []

    for record in records:
        case_summary = record.get("case_summary", "")

        if not isinstance(case_summary, str):
            continue

        for line in case_summary.splitlines():
            if re.match(r"^\s{2,}", line):
                continue

            match = re.match(r"^\s*-?\s*([^:]+):\s*(.*)$", line)
            if not match:
                continue

            rows.append(
                {
                    "prompt_id": record.get("prompt_id"),
                    "breast_cancer_stage": record.get("breast_cancer_stage"),
                    "field": match.group(1).strip(),
                    "value": match.group(2).strip(),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=["prompt_id", "breast_cancer_stage", "field", "value"]
        )

    return pd.DataFrame(rows)


def parse_recommendation_sections(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Parse top-level `Section: value` lines from official recommendations."""
    rows: list[dict[str, Any]] = []

    for record in records:
        recommendations = record.get("official_recommendations", "")

        if not isinstance(recommendations, str):
            continue

        for line in recommendations.splitlines():
            if re.match(r"^\s{2,}", line):
                continue

            match = re.match(r"^\s*-?\s*([^:]+):\s*(.*)$", line)
            if not match:
                continue

            rows.append(
                {
                    "prompt_id": record.get("prompt_id"),
                    "breast_cancer_stage": record.get("breast_cancer_stage"),
                    "section": match.group(1).strip(),
                    "line": line,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=["prompt_id", "breast_cancer_stage", "section", "line"]
        )

    return pd.DataFrame(rows)


def build_schema_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize original input columns."""
    rows: list[dict[str, Any]] = []

    for column in df.columns:
        if column in DERIVED_COLUMNS:
            continue

        if len(df) == 0:
            most_common_type = "unknown"
        else:
            type_counts = (
                df[column].map(lambda value: type(value).__name__).value_counts()
            )
            most_common_type = (
                str(type_counts.index[0]) if len(type_counts) else "unknown"
            )

        rows.append(
            {
                "column": column,
                "non_null": int(df[column].notna().sum()),
                "null": int(df[column].isna().sum()),
                "most_common_type": most_common_type,
            }
        )

    return pd.DataFrame(rows)


def build_summary_tables(
    df: pd.DataFrame,
    rubrics_df: pd.DataFrame,
    tags_df: pd.DataFrame,
    case_fields_df: pd.DataFrame,
    sections_df: pd.DataFrame,
    parse_errors: list[dict[str, Any]],
) -> dict[str, pd.DataFrame]:
    """Build all tabular summaries used by the reports."""
    duplicate_prompt_ids = int(df["prompt_id"].duplicated().sum())
    duplicate_case_summaries = int(df["case_summary"].duplicated().sum())
    duplicate_official_recommendations = int(
        df["official_recommendations"].duplicated().sum()
    )
    duplicate_prompt_contents = int(df["prompt_content"].duplicated().sum())

    rubrics_per_case = df["n_rubrics"].dropna()

    if len(rubrics_per_case):
        median_rubrics_per_case = float(rubrics_per_case.median())
        q1_rubrics_per_case = float(rubrics_per_case.quantile(0.25))
        q3_rubrics_per_case = float(rubrics_per_case.quantile(0.75))
        iqr_rubrics_per_case = q3_rubrics_per_case - q1_rubrics_per_case
        min_rubrics_per_case = int(rubrics_per_case.min())
        max_rubrics_per_case = int(rubrics_per_case.max())
    else:
        median_rubrics_per_case = np.nan
        q1_rubrics_per_case = np.nan
        q3_rubrics_per_case = np.nan
        iqr_rubrics_per_case = np.nan
        min_rubrics_per_case = np.nan
        max_rubrics_per_case = np.nan

    dataset_overview = pd.DataFrame(
        [
            ["records", len(df)],
            ["json_parse_errors", len(parse_errors)],
            [
                "columns_original",
                len([c for c in df.columns if c not in DERIVED_COLUMNS]),
            ],
            ["unique_prompt_ids", int(df["prompt_id"].nunique(dropna=True))],
            ["duplicate_prompt_ids", duplicate_prompt_ids],
            ["duplicate_case_summaries", duplicate_case_summaries],
            [
                "duplicate_official_recommendations",
                duplicate_official_recommendations,
            ],
            ["duplicate_prompt_contents", duplicate_prompt_contents],
            [
                "records_with_single_prompt_message",
                int((df["n_prompt_messages"] == 1).sum()),
            ],
            [
                "records_where_prompt_matches_case_summary",
                int(df["prompt_equals_case_summary"].sum()),
            ],
            [
                "records_where_filename_stage_matches_stage",
                int(df["filename_stage_matches"].sum()),
            ],
            ["rubric_rows", len(rubrics_df)],
            ["median_rubrics_per_case", median_rubrics_per_case],
            ["q1_rubrics_per_case", q1_rubrics_per_case],
            ["q3_rubrics_per_case", q3_rubrics_per_case],
            ["iqr_rubrics_per_case", iqr_rubrics_per_case],
            ["min_rubrics_per_case", min_rubrics_per_case],
            ["max_rubrics_per_case", max_rubrics_per_case],
            ["tag_rows", len(tags_df)],
        ],
        columns=["metric", "value"],
    )

    stage_distribution = (
        df.groupby(["breast_cancer_stage", "filename"], dropna=False)
        .size()
        .reset_index(name="n_cases")
        .sort_values(["breast_cancer_stage", "filename"])
    )

    stage_summary = (
        df.groupby("breast_cancer_stage", dropna=False)
        .agg(
            n_cases=("prompt_id", "count"),
            mean_case_summary_chars=("case_summary_chars", "mean"),
            median_case_summary_chars=("case_summary_chars", "median"),
            mean_recommendations_chars=("official_recommendations_chars", "mean"),
            median_recommendations_chars=("official_recommendations_chars", "median"),
            mean_n_rubrics=("n_rubrics", "mean"),
            median_n_rubrics=("n_rubrics", "median"),
            min_n_rubrics=("n_rubrics", "min"),
            max_n_rubrics=("n_rubrics", "max"),
            mean_total_points=("rubric_total_points", "mean"),
            min_total_points=("rubric_total_points", "min"),
            max_total_points=("rubric_total_points", "max"),
        )
        .round(2)
        .reset_index()
        if len(df)
        else pd.DataFrame()
    )

    if tags_df.empty:
        tag_summary = pd.DataFrame(
            columns=[
                "tag",
                "n_rubrics",
                "n_cases",
                "total_points",
                "mean_points",
                "median_points",
            ]
        )
        tag_stage_summary = pd.DataFrame(
            columns=[
                "breast_cancer_stage",
                "tag",
                "n_rubrics",
                "n_cases",
                "total_points",
                "mean_points",
            ]
        )
    else:
        tag_summary = (
            tags_df.groupby("tag", dropna=False)
            .agg(
                n_rubrics=("criterion", "count"),
                n_cases=("prompt_id", "nunique"),
                total_points=("points", "sum"),
                mean_points=("points", "mean"),
                median_points=("points", "median"),
            )
            .round(2)
            .reset_index()
            .sort_values("n_rubrics", ascending=False)
        )

        tag_stage_summary = (
            tags_df.groupby(["breast_cancer_stage", "tag"], dropna=False)
            .agg(
                n_rubrics=("criterion", "count"),
                n_cases=("prompt_id", "nunique"),
                total_points=("points", "sum"),
                mean_points=("points", "mean"),
            )
            .round(2)
            .reset_index()
            .sort_values(["breast_cancer_stage", "tag"])
        )

    if case_fields_df.empty:
        case_field_summary = pd.DataFrame(
            columns=[
                "field",
                "n_present",
                "n_nao_informado",
                "pct_nao_informado",
                "n_unique_values",
            ]
        )
    else:
        case_field_summary = (
            case_fields_df.groupby("field", dropna=False)
            .agg(
                n_present=("value", "count"),
                n_nao_informado=(
                    "value",
                    lambda values: int(
                        values.astype(str)
                        .str.contains(
                            "Não informado",
                            case=False,
                            regex=False,
                            na=False,
                        )
                        .sum()
                    ),
                ),
                pct_nao_informado=(
                    "value",
                    lambda values: (
                        round(
                            100
                            * values.astype(str)
                            .str.contains(
                                "Não informado",
                                case=False,
                                regex=False,
                                na=False,
                            )
                            .mean(),
                            1,
                        )
                        if len(values)
                        else np.nan
                    ),
                ),
                n_unique_values=("value", "nunique"),
            )
            .reset_index()
            .sort_values(["pct_nao_informado", "field"], ascending=[False, True])
        )

    if sections_df.empty:
        section_summary = pd.DataFrame(
            columns=["section", "n_cases", "n_section_headers"]
        )
    else:
        section_summary = (
            sections_df.groupby("section", dropna=False)
            .agg(n_cases=("prompt_id", "nunique"), n_section_headers=("line", "count"))
            .reset_index()
            .sort_values("n_cases", ascending=False)
        )

    formatting_checks = pd.DataFrame(
        [
            [
                "case_summary_starts_with_dash",
                int(df["case_summary"].astype(str).str.startswith("- ").sum()),
                len(df),
            ],
            [
                "official_recommendations_starts_with_dash",
                int(
                    df["official_recommendations"]
                    .astype(str)
                    .str.startswith("- ")
                    .sum()
                ),
                len(df),
            ],
            [
                "all_rubrics_have_exactly_one_tag",
                int((rubrics_df["n_tags"] == 1).sum()) if len(rubrics_df) else 0,
                len(rubrics_df),
            ],
            ["prompt_role_is_user", int((df["prompt_roles"] == "user").sum()), len(df)],
        ],
        columns=["check", "passing", "total"],
    )
    formatting_checks["pct_passing"] = np.where(
        formatting_checks["total"] > 0,
        (100 * formatting_checks["passing"] / formatting_checks["total"]).round(1),
        np.nan,
    )

    quality_issues = build_quality_issues(df, rubrics_df)
    quality_issues_df = pd.DataFrame({"issue": quality_issues})

    return {
        "dataset_overview": dataset_overview,
        "schema_summary": build_schema_summary(df),
        "stage_distribution": stage_distribution,
        "stage_summary": stage_summary,
        "tag_summary": tag_summary,
        "tag_stage_summary": tag_stage_summary,
        "case_field_summary": case_field_summary,
        "official_recommendation_section_summary": section_summary,
        "formatting_checks": formatting_checks,
        "quality_issues": quality_issues_df,
    }


def build_quality_issues(df: pd.DataFrame, rubrics_df: pd.DataFrame) -> list[str]:
    """Run simple automated quality checks."""
    quality_issues: list[str] = []

    if len(df) == 0:
        return ["The dataset is empty."]

    duplicated_ids = (
        df.loc[df["prompt_id"].duplicated(keep=False), "prompt_id"]
        .dropna()
        .unique()
        .tolist()
    )
    if duplicated_ids:
        quality_issues.append(f"Duplicate prompt_id values: {duplicated_ids[:10]}")

    n_prompt_mismatches = int((~df["prompt_equals_case_summary"]).sum())
    if n_prompt_mismatches:
        quality_issues.append(
            f"{n_prompt_mismatches} records have prompt content that does not exactly "
            "match '# Case summary\\n' + case_summary."
        )

    n_stage_mismatches = int((~df["filename_stage_matches"]).sum())
    if n_stage_mismatches:
        quality_issues.append(
            f"{n_stage_mismatches} records have a breast_cancer_stage that does not "
            "match the stage parsed from filename."
        )

    if len(rubrics_df) and not (rubrics_df["n_tags"] == 1).all():
        quality_issues.append(
            f"{int((rubrics_df['n_tags'] != 1).sum())} rubric items do not have "
            "exactly one tag."
        )

    if len(rubrics_df):
        non_numeric_points = (
            pd.to_numeric(rubrics_df["points"], errors="coerce").isna().sum()
        )
        if non_numeric_points:
            quality_issues.append(
                f"{int(non_numeric_points)} rubric items have non-numeric points."
            )

    n_case_summary_no_dash = len(df) - int(
        df["case_summary"].astype(str).str.startswith("- ").sum()
    )
    if n_case_summary_no_dash:
        quality_issues.append(
            f"{n_case_summary_no_dash} case_summary values do not start with '- '."
        )

    n_recommendations_no_dash = len(df) - int(
        df["official_recommendations"].astype(str).str.startswith("- ").sum()
    )
    if n_recommendations_no_dash:
        quality_issues.append(
            f"{n_recommendations_no_dash} official_recommendations values do not "
            "start with '- '."
        )

    if not quality_issues:
        quality_issues.append(
            "No major quality issue detected by the automated checks."
        )

    return quality_issues


def save_tables(
    output_dir: Path,
    tables: dict[str, pd.DataFrame],
    rubrics_df: pd.DataFrame,
    case_fields_df: pd.DataFrame,
    parse_errors: list[dict[str, Any]],
) -> None:
    """Save all CSV outputs."""
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)

    rubrics_df.to_csv(output_dir / "rubrics_flattened.csv", index=False)
    case_fields_df.to_csv(output_dir / "case_fields_flattened.csv", index=False)

    if parse_errors:
        pd.DataFrame(parse_errors).to_csv(
            output_dir / "json_parse_errors.csv", index=False
        )


def save_bar(
    series: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
    rotation: int = 0,
    figsize: tuple[int, int] = (7, 4),
) -> None:
    """Save a simple bar chart."""
    plt.figure(figsize=figsize)
    series.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation, ha="right" if rotation else "center")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def generate_figures(
    df: pd.DataFrame,
    tag_summary: pd.DataFrame,
    tag_stage_summary: pd.DataFrame,
    case_field_summary: pd.DataFrame,
    figure_dir: Path,
) -> None:
    """Generate EDA figures."""
    figure_dir.mkdir(parents=True, exist_ok=True)

    if len(df):
        save_bar(
            df["breast_cancer_stage"].value_counts(dropna=False).sort_index(),
            "Cases by breast cancer stage",
            "Stage",
            "Number of cases",
            figure_dir / "cases_by_stage.png",
        )

        if df["rubric_total_points"].notna().any():
            plt.figure(figsize=(7, 4))
            df.boxplot(column="rubric_total_points", by="breast_cancer_stage")
            plt.title("Total rubric points by stage")
            plt.suptitle("")
            plt.xlabel("Stage")
            plt.ylabel("Total points per case")
            plt.tight_layout()
            plt.savefig(figure_dir / "total_rubric_points_by_stage.png", dpi=180)
            plt.close()

        if df["n_rubrics"].notna().any():
            plt.figure(figsize=(7, 4))
            df.boxplot(column="n_rubrics", by="breast_cancer_stage")
            plt.title("Number of rubrics per case by stage")
            plt.suptitle("")
            plt.xlabel("Stage")
            plt.ylabel("Number of rubrics")
            plt.tight_layout()
            plt.savefig(figure_dir / "rubrics_per_case_by_stage.png", dpi=180)
            plt.close()

        plt.figure(figsize=(7, 4))
        df[["case_summary_chars", "official_recommendations_chars"]].plot(
            kind="box", ax=plt.gca()
        )
        plt.title("Text length distribution")
        plt.ylabel("Characters")
        plt.tight_layout()
        plt.savefig(figure_dir / "text_length_distribution.png", dpi=180)
        plt.close()

    if not tag_summary.empty:
        save_bar(
            tag_summary.set_index("tag")["n_rubrics"].sort_values(ascending=False),
            "Rubric count by tag",
            "Rubric tag",
            "Number of rubric items",
            figure_dir / "rubric_count_by_tag.png",
            rotation=35,
            figsize=(8, 4),
        )
        save_bar(
            tag_summary.set_index("tag")["total_points"].sort_values(ascending=False),
            "Total rubric points by tag",
            "Rubric tag",
            "Total points",
            figure_dir / "total_points_by_tag.png",
            rotation=35,
            figsize=(8, 4),
        )

    if not tag_stage_summary.empty:
        tag_stage_pivot = tag_stage_summary.pivot(
            index="breast_cancer_stage",
            columns="tag",
            values="total_points",
        ).fillna(0)

        plt.figure(figsize=(8, 4))
        tag_stage_pivot.plot(kind="bar", ax=plt.gca())
        plt.title("Total rubric points by stage and tag")
        plt.xlabel("Stage")
        plt.ylabel("Total points")
        plt.xticks(rotation=0)
        plt.legend(title="Tag", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(figure_dir / "total_points_by_stage_and_tag.png", dpi=180)
        plt.close()

    if not case_field_summary.empty:
        save_bar(
            case_field_summary.set_index("field")["pct_nao_informado"].sort_values(
                ascending=False
            ),
            'Share of "Não informado" by case-summary field',
            "Case-summary field",
            "% with Não informado",
            figure_dir / "nao_informado_by_field.png",
            rotation=45,
            figsize=(9, 4),
        )


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int | None = None) -> str:
    """Convert a DataFrame to a GitHub-flavored Markdown table without tabulate."""
    if df is None or df.empty:
        return "_No rows._"

    if max_rows is not None:
        df = df.head(max_rows)

    display_df = df.copy()
    display_df = display_df.replace({np.nan: ""}).astype(str)

    headers = list(display_df.columns)
    rows = display_df.values.tolist()

    def escape_cell(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", "<br>")

    header_line = "| " + " | ".join(escape_cell(col) for col in headers) + " |"
    separator_line = "| " + " | ".join("---" for _ in headers) + " |"
    row_lines = [
        "| " + " | ".join(escape_cell(str(cell)) for cell in row) + " |" for row in rows
    ]

    return "\n".join([header_line, separator_line, *row_lines])


def build_key_findings(
    df: pd.DataFrame,
    rubrics_df: pd.DataFrame,
    tags_df: pd.DataFrame,
    tag_summary: pd.DataFrame,
    parse_errors: list[dict[str, Any]],
) -> list[str]:
    """Create compact natural-language findings."""
    stage_counts = (
        df["breast_cancer_stage"].value_counts(dropna=False).sort_index().to_dict()
        if len(df)
        else {}
    )
    stage_balanced = len(set(stage_counts.values())) == 1 if stage_counts else False

    findings = [
        f"The dataset contains {len(df)} records and {len(parse_errors)} JSON parsing errors.",
        f"Stage counts: {stage_counts}.",
        "The stage distribution is balanced."
        if stage_balanced
        else "The stage distribution is not perfectly balanced.",
        f"There are {len(rubrics_df)} rubric items and {len(tags_df)} tag assignments.",
    ]

    if len(df) and df["n_rubrics"].notna().any():
        n_rubrics = df["n_rubrics"].dropna()
        q1 = float(n_rubrics.quantile(0.25))
        q3 = float(n_rubrics.quantile(0.75))
        iqr = q3 - q1

        findings.append(
            "The median number of rubrics per case is "
            f"{float(n_rubrics.median()):g} "
            f"(IQR {q1:g}-{q3:g}; range "
            f"{int(n_rubrics.min())}-{int(n_rubrics.max())})."
        )

    if len(rubrics_df):
        findings.append(
            "Every rubric item has exactly one tag."
            if (rubrics_df["n_tags"] == 1).all()
            else "At least one rubric item does not have exactly one tag."
        )

    if not tag_summary.empty:
        top_tag_by_count = tag_summary.sort_values("n_rubrics", ascending=False).iloc[0]
        top_tag_by_points = tag_summary.sort_values(
            "total_points", ascending=False
        ).iloc[0]

        findings.extend(
            [
                f"The most frequent rubric tag is `{top_tag_by_count['tag']}` "
                f"with {int(top_tag_by_count['n_rubrics'])} rubric items.",
                f"The rubric tag with the highest total points is "
                f"`{top_tag_by_points['tag']}` with "
                f"{float(top_tag_by_points['total_points']):g} total points.",
            ]
        )

    duplicate_prompt_ids = int(df["prompt_id"].duplicated().sum()) if len(df) else 0
    findings.append(
        "No duplicate prompt_id values were detected."
        if duplicate_prompt_ids == 0
        else f"{duplicate_prompt_ids} duplicate prompt_id rows were detected."
    )

    if len(df):
        n_prompt_matches = int(df["prompt_equals_case_summary"].sum())
        findings.append(
            "All prompt contents equal '# Case summary' followed by the case_summary field."
            if n_prompt_matches == len(df)
            else f"{len(df) - n_prompt_matches} prompts do not exactly match the case_summary field."
        )

    return findings


def write_reports(
    input_path: Path,
    output_dir: Path,
    tables: dict[str, pd.DataFrame],
    df: pd.DataFrame,
    rubrics_df: pd.DataFrame,
    tags_df: pd.DataFrame,
    parse_errors: list[dict[str, Any]],
) -> tuple[Path, Path]:
    """Write Markdown and HTML reports."""
    figure_dir = output_dir / "figures"
    key_findings = build_key_findings(
        df=df,
        rubrics_df=rubrics_df,
        tags_df=tags_df,
        tag_summary=tables["tag_summary"],
        parse_errors=parse_errors,
    )

    markdown_sections = [
        f"# Exploratory Data Analysis — {input_path.name}",
        "## Key findings",
        "\n".join(f"- {finding}" for finding in key_findings),
        "## Dataset overview",
        dataframe_to_markdown(tables["dataset_overview"]),
        "## Original schema summary",
        dataframe_to_markdown(tables["schema_summary"]),
        "## Stage distribution",
        dataframe_to_markdown(tables["stage_distribution"]),
        "## Stage-level summary",
        dataframe_to_markdown(tables["stage_summary"]),
        "## Rubric tag summary",
        dataframe_to_markdown(tables["tag_summary"]),
        "## Rubric tag by stage",
        dataframe_to_markdown(tables["tag_stage_summary"]),
        "## Case-summary field completeness",
        "This table counts fields parsed from the case summaries and how often each contains `Não informado`.",
        dataframe_to_markdown(tables["case_field_summary"]),
        "## Official recommendation sections",
        dataframe_to_markdown(tables["official_recommendation_section_summary"]),
        "## Formatting checks",
        dataframe_to_markdown(tables["formatting_checks"]),
        "## Automated quality issues",
        dataframe_to_markdown(tables["quality_issues"]),
        "## Generated figures",
        "\n".join(
            f"- `figures/{path.name}`" for path in sorted(figure_dir.glob("*.png"))
        )
        or "_No figures generated._",
        "## Generated data files",
        "\n".join(f"- `{path.name}`" for path in sorted(output_dir.glob("*.csv")))
        or "_No CSV files generated._",
    ]

    report_path = output_dir / "eda_report.md"
    report_path.write_text("\n\n".join(markdown_sections), encoding="utf-8")

    figure_html = "\n".join(
        f"<h3>{path.stem.replace('_', ' ').title()}</h3><img src='figures/{path.name}'>"
        for path in sorted(figure_dir.glob("*.png"))
    )

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>EDA — {input_path.name}</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 40px; line-height: 1.5; max-width: 1100px; }}
table {{ border-collapse: collapse; margin: 16px 0; width: 100%; font-size: 14px; }}
th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
th {{ background: #f6f6f6; }}
img {{ max-width: 900px; width: 100%; border: 1px solid #eee; margin: 12px 0 32px; }}
code {{ background: #f5f5f5; padding: 2px 4px; }}
</style>
</head>
<body>
<h1>Exploratory Data Analysis — {input_path.name}</h1>

<h2>Key findings</h2>
<ul>
{"".join(f"<li>{finding}</li>" for finding in key_findings)}
</ul>

<h2>Dataset overview</h2>
{tables["dataset_overview"].to_html(index=False)}

<h2>Original schema summary</h2>
{tables["schema_summary"].to_html(index=False)}

<h2>Stage distribution</h2>
{tables["stage_distribution"].to_html(index=False)}

<h2>Stage-level summary</h2>
{tables["stage_summary"].to_html(index=False)}

<h2>Rubric tag summary</h2>
{tables["tag_summary"].to_html(index=False)}

<h2>Rubric tag by stage</h2>
{tables["tag_stage_summary"].to_html(index=False)}

<h2>Case-summary field completeness</h2>
<p>This table counts fields parsed from the case summaries and how often each contains <code>Não informado</code>.</p>
{tables["case_field_summary"].to_html(index=False)}

<h2>Official recommendation sections</h2>
{tables["official_recommendation_section_summary"].to_html(index=False)}

<h2>Formatting checks</h2>
{tables["formatting_checks"].to_html(index=False)}

<h2>Automated quality issues</h2>
{tables["quality_issues"].to_html(index=False)}

<h2>Figures</h2>
{figure_html or "<p>No figures generated.</p>"}
</body>
</html>
"""

    html_path = output_dir / "eda_report.html"
    html_path.write_text(html, encoding="utf-8")

    return report_path, html_path


def zip_output_dir(output_dir: Path, zip_path: Path | None = None) -> Path:
    """Create a ZIP archive with all generated outputs."""
    if zip_path is None:
        zip_path = output_dir.with_suffix(".zip")

    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for file_path in output_dir.rglob("*"):
            archive.write(file_path, file_path.relative_to(output_dir.parent))

    return zip_path


def run_eda(
    input_path: Path,
    output_dir: Path,
    overwrite: bool = False,
    create_zip: bool = False,
) -> dict[str, Path | int]:
    """Run the complete EDA workflow."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.is_dir():
        raise IsADirectoryError(
            f"Input path is a directory, expected JSONL file: {input_path}"
        )

    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    records, parse_errors = load_jsonl(input_path)
    df = pd.DataFrame(records)

    expected_columns = [
        "prompt_id",
        "prompt",
        "case_summary",
        "official_recommendations",
        "breast_cancer_stage",
        "filename",
        "rubrics",
    ]
    df = ensure_columns(df, expected_columns)
    df = add_derived_columns(df)

    rubrics_df = flatten_rubrics(records)
    tags_df = flatten_tags(rubrics_df)
    case_fields_df = parse_case_summary_fields(records)
    sections_df = parse_recommendation_sections(records)

    tables = build_summary_tables(
        df=df,
        rubrics_df=rubrics_df,
        tags_df=tags_df,
        case_fields_df=case_fields_df,
        sections_df=sections_df,
        parse_errors=parse_errors,
    )

    save_tables(
        output_dir=output_dir,
        tables=tables,
        rubrics_df=rubrics_df,
        case_fields_df=case_fields_df,
        parse_errors=parse_errors,
    )

    generate_figures(
        df=df,
        tag_summary=tables["tag_summary"],
        tag_stage_summary=tables["tag_stage_summary"],
        case_field_summary=tables["case_field_summary"],
        figure_dir=figure_dir,
    )

    markdown_report, html_report = write_reports(
        input_path=input_path,
        output_dir=output_dir,
        tables=tables,
        df=df,
        rubrics_df=rubrics_df,
        tags_df=tags_df,
        parse_errors=parse_errors,
    )

    result: dict[str, Path | int] = {
        "records": len(df),
        "parse_errors": len(parse_errors),
        "output_dir": output_dir,
        "markdown_report": markdown_report,
        "html_report": html_report,
    }

    if create_zip:
        result["zip_path"] = zip_output_dir(output_dir)

    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run exploratory data analysis for an evaluation JSONL dataset."
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
        help="Path to the input JSONL evaluation dataset.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="Directory where reports, CSVs, and figures will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory first if it already exists.",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Also create a ZIP archive next to the output directory.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)

    try:
        result = run_eda(
            input_path=args.input,
            output_dir=args.output,
            overwrite=args.overwrite,
            create_zip=args.zip,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("EDA completed successfully.")
    print(f"Records: {result['records']}")
    print(f"JSON parse errors: {result['parse_errors']}")
    print(f"Output directory: {result['output_dir']}")
    print(f"Markdown report: {result['markdown_report']}")
    print(f"HTML report: {result['html_report']}")

    if "zip_path" in result:
        print(f"ZIP archive: {result['zip_path']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
