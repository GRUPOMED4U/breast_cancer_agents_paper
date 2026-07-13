from __future__ import annotations

from pathlib import Path
from typing import Literal, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from logs import get_logger

logger = get_logger(__name__)


def scatter_plot(
    data: pd.DataFrame,
    x: str,
    y: str,
    *,
    hue: Optional[str] = None,
    style: Optional[str] = None,
    size: Optional[str] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
    log_x: bool = False,
    log_y: bool = False,
    alpha: float = 0.85,
    point_size: int = 85,
    add_trendline: bool = True,
    trendline_label: Optional[str] = "Linear trend",
    annotate: Optional[Sequence[str]] = None,
    annotate_fontsize: int = 8,
    figsize: tuple[float, float] = (11.5, 6.5),
    save_path: Optional[str | Path] = None,
    dpi: int = 300,
    hue_label_map: Optional[Mapping[str, str]] = None,
    style_label_map: Optional[Mapping[str, str]] = None,
    style_marker_map: Optional[Mapping[str, str]] = None,
    hue_color_map: Optional[Mapping[str, str]] = None,
    hue_order: Optional[Sequence[str]] = None,
    style_order: Optional[Sequence[str]] = None,
    hue_legend_title: Optional[str] = None,
    style_legend_title: Optional[str] = None,
    legend_x: float = 1.02,
    legend_width: float = 0.48,
    legend_right_margin: float = 0.56,
    legend_top_y: float = 0.90,
    legend_vertical_gap: float = 0.22,
    legend_marker_size: int = 8,
    show_legend: bool = True,
    show_grid: bool = True,
    quadrant_reference: bool = False,
    reference_x: Optional[float] = None,
    reference_y: Optional[float] = None,
    reference_label: Optional[str] = "Reference",
):
    """
    Create a publication-ready scatter plot from a pandas DataFrame.

    Common use case
    ---------------
    - hue="model_id" controls color.
    - style="pipeline_id" controls marker shape.

    Parameters
    ----------
    log_x:
        If True, use a logarithmic scale for the x-axis.

    log_y:
        If True, use a logarithmic scale for the y-axis.

    hue_legend_title:
        Custom title for the hue/color legend.
        If None, uses the hue column name.

    style_legend_title:
        Custom title for the style/marker legend.
        If None, uses the style column name.
    """

    df = data.copy()

    required_cols = [x, y]
    if hue:
        required_cols.append(hue)
    if style:
        required_cols.append(style)
    if size:
        required_cols.append(size)

    df = df.dropna(subset=required_cols)

    if df.empty:
        raise ValueError("No rows left after dropping missing values.")

    df[x] = df[x].astype(float)
    df[y] = df[y].astype(float)

    if log_x and (df[x] <= 0).any():
        raise ValueError(
            f"`log_x=True` requires all values in column {x!r} to be positive."
        )

    if log_y and (df[y] <= 0).any():
        raise ValueError(
            f"`log_y=True` requires all values in column {y!r} to be positive."
        )

    if xlim is not None and log_x and (xlim[0] <= 0 or xlim[1] <= 0):
        raise ValueError("`xlim` values must be positive when `log_x=True`.")

    if ylim is not None and log_y and (ylim[0] <= 0 or ylim[1] <= 0):
        raise ValueError("`ylim` values must be positive when `log_y=True`.")

    if hue:
        df[hue] = df[hue].astype(str)
    if style:
        df[style] = df[style].astype(str)

    hue_label_map = dict(hue_label_map or {})
    style_label_map = dict(style_label_map or {})
    style_marker_map = dict(style_marker_map or {})
    hue_color_map = dict(hue_color_map or {})

    def rename_hue_label(value: object) -> str:
        value = str(value)
        return hue_label_map.get(value, value)

    def rename_style_label(value: object) -> str:
        value = str(value)
        return style_label_map.get(value, value)

    fig, ax = plt.subplots(figsize=figsize)

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")

    if log_x:
        ax.set_xscale("log")

    if log_y:
        ax.set_yscale("log")

    if show_grid:
        ax.grid(
            True,
            which="major",
            linestyle="-",
            linewidth=0.5,
            alpha=0.18,
            color="#888888",
        )

        if log_x or log_y:
            ax.grid(
                True,
                which="minor",
                linestyle="-",
                linewidth=0.35,
                alpha=0.08,
                color="#888888",
            )

        ax.set_axisbelow(True)

    # ------------------------------------------------------------------
    # Point sizes
    # ------------------------------------------------------------------
    if size:
        size_values = df[size].astype(float)

        if size_values.max() == size_values.min():
            df["_plot_size"] = float(point_size)
        else:
            df["_plot_size"] = 50 + 250 * (size_values - size_values.min()) / (
                size_values.max() - size_values.min()
            )
    else:
        df["_plot_size"] = float(point_size)

    # ------------------------------------------------------------------
    # Hue color mapping
    # ------------------------------------------------------------------
    if hue:
        if hue_order is None:
            hue_values = df[hue].drop_duplicates().to_numpy()
        else:
            hue_values = [
                str(value) for value in hue_order if str(value) in set(df[hue])
            ]

        default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        for i, hue_value in enumerate(hue_values):
            hue_color_map.setdefault(
                str(hue_value),
                default_colors[i % len(default_colors)],
            )
    else:
        hue_values = [None]
        hue_color_map = {None: "#333333"}

    # ------------------------------------------------------------------
    # Style marker mapping
    # ------------------------------------------------------------------
    default_markers = [
        "o",
        "s",
        "^",
        "D",
        "P",
        "X",
        "v",
        "<",
        ">",
        "*",
        "h",
        "8",
        "p",
    ]

    if style:
        if style_order is None:
            style_values = df[style].drop_duplicates().to_numpy()
        else:
            style_values = [
                str(value) for value in style_order if str(value) in set(df[style])
            ]

        missing_style_values = [
            str(value) for value in style_values if str(value) not in style_marker_map
        ]

        if len(missing_style_values) > len(default_markers):
            raise ValueError(
                f"Column {style!r} has {len(missing_style_values)} style values "
                "without assigned markers, but there are not enough default markers. "
                "Please provide `style_marker_map` manually."
            )

        used_markers = set(style_marker_map.values())
        available_markers = [
            marker for marker in default_markers if marker not in used_markers
        ]

        for style_value, marker in zip(missing_style_values, available_markers):
            style_marker_map[str(style_value)] = marker
    else:
        style_values = [None]

    # ------------------------------------------------------------------
    # Scatter plot
    # ------------------------------------------------------------------
    for hue_value in hue_values:
        for style_value in style_values:
            subset = df.copy()

            if hue:
                subset = subset[subset[hue] == str(hue_value)]

            if style:
                subset = subset[subset[style] == str(style_value)]
                marker = style_marker_map[str(style_value)]
            else:
                marker = "o"

            if subset.empty:
                continue

            ax.scatter(
                subset[x],
                subset[y],
                s=subset["_plot_size"],
                alpha=alpha,
                marker=marker,
                color=hue_color_map[str(hue_value)] if hue else "#333333",
                edgecolor="white",
                linewidth=0.9,
                zorder=3,
            )

    # ------------------------------------------------------------------
    # Reference lines
    # ------------------------------------------------------------------
    if quadrant_reference:
        reference_x = float(df[x].median()) if reference_x is None else reference_x
        reference_y = float(df[y].median()) if reference_y is None else reference_y

        if log_x and reference_x <= 0:
            raise ValueError("`reference_x` must be positive when `log_x=True`.")

        if log_y and reference_y <= 0:
            raise ValueError("`reference_y` must be positive when `log_y=True`.")

        ax.axvline(
            reference_x,
            color="#777777",
            linewidth=1.0,
            linestyle="--",
            alpha=0.55,
            zorder=1,
        )

        ax.axhline(
            reference_y,
            color="#777777",
            linewidth=1.0,
            linestyle="--",
            alpha=0.55,
            zorder=1,
        )

        if reference_label:
            ax.text(
                reference_x,
                reference_y,
                f" {reference_label}",
                ha="left",
                va="bottom",
                fontsize=9,
                color="#555555",
                alpha=0.85,
            )

    # ------------------------------------------------------------------
    # Trendline
    # ------------------------------------------------------------------
    if add_trendline and len(df) >= 2:
        x_values = df[x].astype(float).to_numpy()
        y_values = df[y].astype(float).to_numpy()

        if log_x:
            x_fit = np.log10(x_values)
            x_line = np.logspace(
                np.log10(x_values.min()),
                np.log10(x_values.max()),
                200,
            )
            x_line_fit = np.log10(x_line)
        else:
            x_fit = x_values
            x_line = np.linspace(x_values.min(), x_values.max(), 200)
            x_line_fit = x_line

        if log_y:
            y_fit = np.log10(y_values)
        else:
            y_fit = y_values

        coef = np.polyfit(x_fit, y_fit, deg=1)
        poly = np.poly1d(coef)

        y_line_fit = poly(x_line_fit)

        if log_y:
            y_line = 10**y_line_fit
        else:
            y_line = y_line_fit

        ax.plot(
            x_line,
            y_line,
            linewidth=2.0,
            alpha=0.85,
            linestyle="-",
            color="#333333",
            label=trendline_label,
            zorder=2,
        )

    # ------------------------------------------------------------------
    # Annotations
    # ------------------------------------------------------------------
    if annotate is not None:
        for label, (_, row) in zip(annotate, df.iterrows()):
            ax.annotate(
                str(label),
                xy=(row[x], row[y]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=annotate_fontsize,
                color="#333333",
                alpha=0.9,
            )

    # ------------------------------------------------------------------
    # Legends
    # ------------------------------------------------------------------
    if show_legend:
        if hue:
            hue_handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor=hue_color_map[str(hue_value)],
                    markeredgecolor="white",
                    markersize=legend_marker_size,
                    linestyle="none",
                    label=rename_hue_label(hue_value),
                )
                for hue_value in hue_values
            ]

            hue_legend = ax.legend(
                handles=hue_handles,
                title=hue_legend_title or hue,
                loc="upper left",
                bbox_to_anchor=(legend_x, legend_top_y, legend_width, 0.1),
                bbox_transform=ax.transAxes,
                borderaxespad=0,
                frameon=True,
                framealpha=1.0,
                edgecolor="#CCCCCC",
                fancybox=False,
                mode="expand",
                fontsize=9,
                title_fontsize=10,
                handletextpad=0.8,
                borderpad=0.8,
                labelspacing=0.55,
            )

            ax.add_artist(hue_legend)

        if style:
            style_handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker=style_marker_map[str(style_value)],
                    color="black",
                    markerfacecolor="black",
                    markeredgecolor="white",
                    markersize=legend_marker_size,
                    linestyle="none",
                    label=rename_style_label(style_value),
                )
                for style_value in style_values
            ]

            style_legend_y = legend_top_y - legend_vertical_gap if hue else legend_top_y

            ax.legend(
                handles=style_handles,
                title=style_legend_title or style,
                loc="upper left",
                bbox_to_anchor=(legend_x, style_legend_y, legend_width, 0.1),
                bbox_transform=ax.transAxes,
                borderaxespad=0,
                frameon=True,
                framealpha=1.0,
                edgecolor="#CCCCCC",
                fancybox=False,
                mode="expand",
                fontsize=9,
                title_fontsize=10,
                handletextpad=0.8,
                borderpad=0.8,
                labelspacing=0.55,
            )

    # ------------------------------------------------------------------
    # Labels and limits
    # ------------------------------------------------------------------
    if title:
        if subtitle:
            ax.set_title(
                f"{title}\n{subtitle}",
                fontsize=15,
                fontweight="bold",
                pad=14,
                loc="left",
            )
        else:
            ax.set_title(
                title,
                fontsize=15,
                fontweight="bold",
                pad=14,
                loc="left",
            )

    ax.set_xlabel(xlabel or x, fontsize=12, labelpad=10)
    ax.set_ylabel(ylabel or y, fontsize=12, labelpad=10)

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=10,
        colors="#222222",
        width=0.8,
        length=4,
    )

    ax.tick_params(
        axis="both",
        which="minor",
        colors="#222222",
        width=0.6,
        length=2.5,
    )

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.margins(x=0.04, y=0.08)

    if show_legend and (hue or style):
        fig.subplots_adjust(right=legend_right_margin)
    else:
        fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, ax


def bar_plot(
    data: pd.DataFrame,
    x: str,
    y: str,
    *,
    hue: Optional[str] = None,
    error: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    x_label_map: Optional[Mapping[str, str]] = None,
    hue_label_map: Optional[Mapping[str, str]] = None,
    ylabel: Optional[str] = None,
    ylim: Optional[tuple[float, float]] = None,
    horizontal: bool = False,
    sort_by_y: bool = False,
    ascending: bool = False,
    sort_hue: Optional[str] = None,
    sort_agg: str = "mean",
    show_values: bool = True,
    value_format: str = ".3f",
    legend_outside: bool = True,
    bar_width: float = 0.75,
    figsize: tuple[float, float] = (11, 6),
    save_path: Optional[str | Path] = None,
    dpi: int = 300,
    value_label_padding: float = 0.03,
    legend_title: Optional[str] = None,
    color: None | str | list[str] | tuple[str, ...] | Mapping[str, str] = None,
):
    """
    Create a polished bar plot from a pandas DataFrame.

    Parameters
    ----------
    color:
        - None: use Matplotlib default colors.
        - str: use the same color for all bars.
        - list/tuple: when hue is used, assign one color per hue in hue order.
        - Mapping[str, str]: when hue is used, map original hue values to colors.

    Examples
    --------
    Use specific colors for hue categories:

        bar_plot(
            df,
            x="pipeline_id",
            y="global_score",
            hue="model_id",
            color={
                "gemini-gemini-3.5-flash": "#E69F00",
                "gemini-gemini-3.1-pro-preview": "#56B4E9",
                "gemini-gemini-2.5-flash": "#009E73",
            },
        )
    """

    required_cols = [x, y]
    if hue:
        required_cols.append(hue)
    if error:
        required_cols.append(error)

    df = data.dropna(subset=required_cols).copy()

    df[x] = df[x].astype(str)
    if hue:
        df[hue] = df[hue].astype(str)

    x_label_map = dict(x_label_map or {})
    hue_label_map = dict(hue_label_map or {})

    def rename_x_label(value: object) -> str:
        value = str(value)
        return x_label_map.get(value, value)

    def rename_hue_label(value: object) -> str:
        value = str(value)
        return hue_label_map.get(value, value)

    def get_hue_color(hue_value: str, hue_index: int):
        """
        Resolve color for a specific hue category.
        """
        if color is None:
            return None

        if isinstance(color, Mapping):
            return color.get(str(hue_value))

        if isinstance(color, str):
            return color

        if isinstance(color, (list, tuple)):
            if len(color) == 0:
                return None
            return color[hue_index % len(color)]

        return None

    if hue is None and isinstance(color, Mapping):
        raise ValueError(
            "`color` as a dictionary/mapping only works when `hue` is provided."
        )

    # ---------------------------------------------------------------------
    # Category order
    # ---------------------------------------------------------------------
    if sort_by_y:
        if hue is None:
            sort_values = (
                df.groupby(x, observed=True)[y]
                .agg(sort_agg)
                .sort_values(ascending=ascending)
            )
            category_order = sort_values.index.to_numpy()

        else:
            if sort_hue is not None:
                sort_df = df[df[hue] == str(sort_hue)]

                if sort_df.empty:
                    raise ValueError(
                        f"sort_hue={sort_hue!r} was not found in column {hue!r}. "
                        f"Available values are: {sorted(df[hue].unique())}"
                    )

                sort_values = (
                    sort_df.groupby(x, observed=True)[y]
                    .agg(sort_agg)
                    .sort_values(ascending=ascending)
                )

            else:
                sort_values = (
                    df.groupby(x, observed=True)[y]
                    .agg(sort_agg)
                    .sort_values(ascending=ascending)
                )

            category_order = sort_values.index.to_numpy()

    else:
        category_order = df[x].drop_duplicates().to_numpy()

    # ---------------------------------------------------------------------
    # Hue order
    # ---------------------------------------------------------------------
    if hue:
        hue_values = df[hue].drop_duplicates().to_numpy()

    # ---------------------------------------------------------------------
    # Figure setup
    # ---------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.spines["left"].set_alpha(0.25)
    ax.spines["bottom"].set_alpha(0.25)

    ax.grid(
        True,
        axis="x" if horizontal else "y",
        linestyle="--",
        linewidth=0.7,
        alpha=0.25,
    )

    # ---------------------------------------------------------------------
    # No hue
    # ---------------------------------------------------------------------
    if hue is None:
        plot_df = df.groupby(x, observed=True, as_index=False).agg(
            {y: sort_agg, **({error: sort_agg} if error else {})}
        )

        plot_df = plot_df.set_index(x).reindex(category_order)

        categories = plot_df.index.to_numpy()
        values = plot_df[y].astype(float).to_numpy()
        errors = plot_df[error].astype(float).to_numpy() if error else None
        positions = np.arange(len(categories))

        if horizontal:
            bars = ax.barh(
                positions,
                values,
                xerr=errors,
                height=bar_width,
                alpha=0.85,
                edgecolor="white",
                linewidth=0.8,
                capsize=4 if error else 0,
                color=color,
            )
            ax.set_yticks(positions)
            ax.set_yticklabels([rename_x_label(category) for category in categories])

            # Makes category_order[0] appear at the top.
            ax.invert_yaxis()

        else:
            bars = ax.bar(
                positions,
                values,
                yerr=errors,
                width=bar_width,
                alpha=0.85,
                edgecolor="white",
                linewidth=0.8,
                capsize=4 if error else 0,
                color=color,
            )
            ax.set_xticks(positions)
            ax.set_xticklabels(
                [rename_x_label(category) for category in categories],
                rotation=30,
                ha="right",
            )

        if show_values:
            for bar, value in zip(bars, values):
                if np.isnan(value):
                    continue

                label = format(value, value_format)

                if horizontal:
                    ax.text(
                        bar.get_width() + value_label_padding,
                        bar.get_y() + bar.get_height() / 2,
                        f" {label}",
                        va="center",
                        fontsize=9,
                        alpha=0.85,
                    )
                else:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + value_label_padding,
                        label,
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        alpha=0.85,
                    )

    # ---------------------------------------------------------------------
    # With hue
    # ---------------------------------------------------------------------
    else:
        categories = category_order
        x_positions = np.arange(len(categories))
        n_hues = len(hue_values)
        single_bar_width = bar_width / n_hues

        for i, hue_value in enumerate(hue_values):
            subset = df[df[hue] == hue_value].copy()

            agg_dict = {y: sort_agg}
            if error:
                agg_dict[error] = sort_agg

            subset = (
                subset.groupby(x, observed=True, as_index=False)
                .agg(agg_dict)
                .set_index(x)
                .reindex(categories)
            )

            values = subset[y].astype(float).to_numpy()
            errors = subset[error].astype(float).to_numpy() if error else None

            offset = (i - (n_hues - 1) / 2) * single_bar_width
            positions = x_positions + offset

            bar_color = get_hue_color(str(hue_value), i)

            if horizontal:
                bars = ax.barh(
                    positions,
                    values,
                    xerr=errors,
                    height=single_bar_width,
                    alpha=0.85,
                    edgecolor="white",
                    linewidth=0.8,
                    label=rename_hue_label(hue_value),
                    capsize=4 if error else 0,
                    color=bar_color,
                )
            else:
                bars = ax.bar(
                    positions,
                    values,
                    yerr=errors,
                    width=single_bar_width,
                    alpha=0.85,
                    edgecolor="white",
                    linewidth=0.8,
                    label=rename_hue_label(hue_value),
                    capsize=4 if error else 0,
                    color=bar_color,
                )

            if show_values:
                for bar, value in zip(bars, values):
                    if np.isnan(value):
                        continue

                    label = format(value, value_format)

                    if horizontal:
                        ax.text(
                            bar.get_width() + value_label_padding,
                            bar.get_y() + bar.get_height() / 2,
                            f" {label}",
                            va="center",
                            fontsize=8,
                            alpha=0.85,
                        )
                    else:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + value_label_padding,
                            label,
                            ha="center",
                            va="bottom",
                            fontsize=8,
                            alpha=0.85,
                        )

        if horizontal:
            ax.set_yticks(x_positions)
            ax.set_yticklabels([rename_x_label(category) for category in categories])

            # Critical for intuitive top-to-bottom sorting.
            ax.invert_yaxis()

        else:
            ax.set_xticks(x_positions)
            ax.set_xticklabels(
                [rename_x_label(category) for category in categories],
                rotation=30,
                ha="right",
            )

        if legend_outside:
            ax.legend(
                title=legend_title or hue,
                loc="upper left",
                bbox_to_anchor=(1.02, 1),
                borderaxespad=0,
                frameon=True,
                framealpha=0.95,
            )
            fig.subplots_adjust(right=0.72)
        else:
            ax.legend(
                title=legend_title or hue,
                frameon=True,
                framealpha=0.95,
            )

    # ---------------------------------------------------------------------
    # Labels and limits
    # ---------------------------------------------------------------------
    if title:
        ax.set_title(
            title or f"{y} by {x}",
            fontsize=16,
            fontweight="bold",
            pad=14,
        )

    if horizontal:
        ax.set_xlabel(ylabel or y, fontsize=12, labelpad=10)
        ax.set_ylabel(xlabel or x, fontsize=12, labelpad=10)

        if ylim is not None:
            ax.set_xlim(ylim)
    else:
        ax.set_xlabel(xlabel or x, fontsize=12, labelpad=10)
        ax.set_ylabel(ylabel or y, fontsize=12, labelpad=10)

        if ylim is not None:
            ax.set_ylim(ylim)

    ax.tick_params(axis="both", labelsize=10)

    if not (hue and legend_outside):
        plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, ax


def radar_plot_relative_to_baseline(
    data: pd.DataFrame,
    *,
    model_col: str = "model_id",
    pipeline_col: str = "pipeline_id",
    value_col: str = "global_score",
    baseline_pipeline: str = "single_llm_zero_shot",
    pipeline_order: Optional[Sequence[str]] = None,
    pipeline_label_map: Optional[Mapping[str, str]] = None,
    model_order: Optional[Sequence[str]] = None,
    model_label_map: Optional[Mapping[str, str]] = None,
    relative_mode: Literal["ratio", "delta", "percent_change"] = "ratio",
    title: Optional[str] = None,
    figsize: tuple[float, float] = (9, 9),
    ylim: Optional[tuple[float, float]] = None,
    show_baseline_circle: bool = True,
    fill_alpha: float = 0.08,
    linewidth: float = 2.2,
    marker_size: float = 5,
    legend_title: str = "Model",
    legend_bbox_to_anchor: tuple[float, float] = (1.25, 1.05),
    save_path: Optional[str | Path] = None,
    dpi: int = 300,
    grid_label_alpha: float = 0.45,
    grid_label_suffix: str = "%",
):
    """
    Create a radar plot comparing each model's pipeline performance relative
    to that model's own baseline pipeline.

    Values are normalized within each model, using `baseline_pipeline` as the
    reference.

    Parameters
    ----------
    data:
        Input DataFrame containing model, pipeline, and score columns.

    model_col:
        Column identifying the model.

    pipeline_col:
        Column identifying the pipeline.

    value_col:
        Column containing the metric to plot.

    baseline_pipeline:
        Pipeline used as the within-model reference.

    pipeline_order:
        Optional explicit order of radar axes. Use the original pipeline IDs,
        not the display labels.

    pipeline_label_map:
        Optional mapping from original pipeline IDs to concise display labels.
        Pipelines not present in the mapping keep their original name.

        Example:
            {
                "single_llm_zero_shot": "ZS",
                "single_llm_with_web_search": "Web",
                "single_llm_with_pubmed_search": "PubMed",
                "divide_and_conquer": "D&C",
                "divide_and_conquer_with_subagents_auto_spawning": "D&C+Agents",
            }

    model_order:
        Optional explicit order of model traces.

    relative_mode:
        How to compare each pipeline to the baseline.

        - "ratio":
            value / baseline
            1.0 means equal to baseline.

        - "delta":
            value - baseline
            0.0 means equal to baseline.

        - "percent_change":
            100 * (value - baseline) / baseline
            0.0 means equal to baseline.

    title:
        Plot title.

    figsize:
        Figure size.

    ylim:
        Radial axis limits. If None, inferred from the data.

    show_baseline_circle:
        Whether to draw the reference circle corresponding to baseline
        performance.

    fill_alpha:
        Transparency for the filled radar polygons.

    linewidth:
        Width of the model lines.

    marker_size:
        Size of markers on radar vertices.

    legend_title:
        Legend title.

    legend_bbox_to_anchor:
        Location of the legend outside the plot.

    save_path:
        Optional path to save the figure.

    dpi:
        Resolution used when saving.

    grid_label_alpha:
        Transparency for grid labels.

    grid_label_suffix:
        Suffix for grid labels.

    Returns
    -------
    fig, ax:
        Matplotlib figure and polar axis.
    """

    required_cols = {model_col, pipeline_col, value_col}
    missing_cols = required_cols - set(data.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    pipeline_label_map = dict(pipeline_label_map or {})
    model_label_map = dict(model_label_map or {})

    df = data[[model_col, pipeline_col, value_col]].copy()

    baseline_df = df[df[pipeline_col] == baseline_pipeline][
        [model_col, value_col]
    ].rename(columns={value_col: "_baseline_value"})

    if baseline_df.empty:
        raise ValueError(
            f"Baseline pipeline {baseline_pipeline!r} was not found in "
            f"column {pipeline_col!r}."
        )

    duplicated_baselines = baseline_df[model_col].duplicated()
    if duplicated_baselines.any():
        duplicated_models = baseline_df.loc[duplicated_baselines, model_col].tolist()
        raise ValueError(
            "Each model must have exactly one baseline row. "
            f"Duplicated baselines found for: {duplicated_models}"
        )

    df = df.merge(baseline_df, on=model_col, how="left")

    # drop baseline rows
    df = df[df[pipeline_col] != baseline_pipeline]

    missing_baseline_models = df.loc[df["_baseline_value"].isna(), model_col].unique()
    if len(missing_baseline_models) > 0:
        raise ValueError(
            f"Some models do not have a baseline row: {list(missing_baseline_models)}"
        )

    if (df["_baseline_value"] == 0).any() and relative_mode in {
        "ratio",
        "percent_change",
    }:
        zero_models = df.loc[df["_baseline_value"] == 0, model_col].unique()
        raise ValueError(
            "Cannot compute ratio or percent change when the baseline is zero. "
            f"Models with zero baseline: {list(zero_models)}"
        )

    if relative_mode == "ratio":
        df["_relative_value"] = df[value_col] / df["_baseline_value"]
        reference_value = 1.0
        radial_label = f"{value_col} / baseline"

    elif relative_mode == "delta":
        df["_relative_value"] = df[value_col] - df["_baseline_value"]
        reference_value = 0.0
        radial_label = f"{value_col} difference vs. baseline"

    elif relative_mode == "percent_change":
        df["_relative_value"] = (
            100 * (df[value_col] - df["_baseline_value"]) / df["_baseline_value"]
        )
        reference_value = 0.0
        radial_label = "% change vs. baseline"

    else:
        raise ValueError(
            "relative_mode must be one of: 'ratio', 'delta', or 'percent_change'."
        )

    if pipeline_order is None:
        pipelines = sorted(df[pipeline_col].unique())

        if baseline_pipeline in pipelines:
            pipelines = [
                pipeline for pipeline in pipelines if pipeline != baseline_pipeline
            ]
    else:
        pipelines = [
            pipeline for pipeline in pipeline_order if pipeline != baseline_pipeline
        ]

    if len(pipelines) < 3:
        raise ValueError(
            "Radar plots need at least 3 pipelines/axes to be informative."
        )

    pipeline_labels = [
        pipeline_label_map.get(pipeline, pipeline) for pipeline in pipelines
    ]

    duplicated_labels = pd.Series(pipeline_labels).duplicated()
    if duplicated_labels.any():
        duplicated_display_labels = (
            pd.Series(pipeline_labels).loc[duplicated_labels].tolist()
        )
        raise ValueError(
            "Pipeline display labels must be unique. "
            f"Duplicated labels found: {duplicated_display_labels}"
        )

    if model_order is None:
        models = sorted(df[model_col].unique())
    else:
        models = list(model_order)

    radar_df = df.pivot_table(
        index=model_col,
        columns=pipeline_col,
        values="_relative_value",
        aggfunc="mean",
    ).reindex(index=models, columns=pipelines)

    if radar_df.isna().any().any():
        missing = radar_df.isna()
        missing_pairs = [
            (model, pipeline)
            for model in radar_df.index
            for pipeline in radar_df.columns
            if missing.loc[model, pipeline]
        ]
        logger.warning(f"Some model/pipeline combinations are missing: {missing_pairs}")

    values = radar_df.to_numpy(dtype=float)

    n_axes = len(pipelines)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False)
    closed_angles = np.concatenate([angles, [angles[0]]])

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for model in radar_df.index:
        model_values = radar_df.loc[model].to_numpy(dtype=float)
        closed_values = np.concatenate([model_values, [model_values[0]]])
        model_label = model_label_map.get(str(model), str(model))

        ax.plot(
            closed_angles,
            closed_values,
            linewidth=linewidth,
            marker="o",
            markersize=marker_size,
            label=model_label,
        )
        ax.fill(
            closed_angles,
            closed_values,
            alpha=fill_alpha,
        )

    if show_baseline_circle:
        ax.plot(
            closed_angles,
            np.full_like(closed_angles, reference_value, dtype=float),
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            label="Baseline reference",
        )

    ax.set_xticks(angles)
    ax.set_xticklabels(pipeline_labels)

    if ylim is None:
        min_value = np.nanmin(values)
        max_value = np.nanmax(values)

        if relative_mode == "ratio":
            lower = min(0.8, min_value - 0.05)
            upper = max(1.2, max_value + 0.05)
        elif relative_mode == "delta":
            padding = max(0.03, 0.15 * (max_value - min_value))
            lower = min(min_value - padding, reference_value - padding)
            upper = max(max_value + padding, reference_value + padding)
        else:
            padding = max(5.0, 0.15 * (max_value - min_value))
            lower = min(min_value - padding, reference_value - padding)
            upper = max(max_value + padding, reference_value + padding)

        ylim = (lower, upper)

    ax.set_ylim(*ylim)

    yticks = ax.get_yticks()

    if relative_mode == "percent_change":
        ytick_labels = [f"{tick:.0f}{grid_label_suffix}" for tick in yticks]
    elif relative_mode == "ratio":
        ytick_labels = [f"{tick * 100:.0f}{grid_label_suffix}" for tick in yticks]
    else:
        ytick_labels = [f"{tick:.2f}" for tick in yticks]

    ax.set_yticklabels(ytick_labels)
    for label in ax.get_yticklabels():
        label.set_alpha(grid_label_alpha)

    ax.set_rlabel_position(90)
    ax.grid(True, alpha=0.35)
    ax.spines["polar"].set_alpha(0.25)

    if title:
        ax.set_title(title, pad=30, fontsize=14, fontweight="bold")

    ax.text(
        0.5,
        -0.08,
        radial_label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
    )

    ax.legend(
        title=legend_title,
        loc="upper right",
        bbox_to_anchor=legend_bbox_to_anchor,
        frameon=True,
    )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, ax
