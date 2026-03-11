from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Literal


@dataclass(frozen=True)
class PlotMetricSpec:
    metric_dir: str
    label: str
    value_field: str
    std_field: str | None


@dataclass(frozen=True)
class PlotSeries:
    metric_dir: str
    label: str
    values_by_model: dict[str, float]
    std_by_model: dict[str, float]


PlotGroupBy = Literal["metric", "model"]


@dataclass(frozen=True)
class GroupedPlotData:
    group_labels: list[str]
    bar_labels: list[str]
    values_by_bar: list[list[float]]
    std_by_bar: list[list[float]]
    x_axis_label: str
    legend_title: str


@dataclass(frozen=True)
class MetricPlotData:
    metric_dir: str
    label: str
    slug: str
    model_labels: list[str]
    values: list[float]
    std_values: list[float]


@dataclass(frozen=True)
class RenderedPlotOutputs:
    combined_png: Path
    combined_svg: Path
    metric_pngs: dict[str, Path]
    metric_svgs: dict[str, Path]


PLOT_METRICS = (
    PlotMetricSpec(
        metric_dir="ctc",
        label="CTC Closeness",
        value_field="metric_mean",
        std_field="metric_std",
    ),
    PlotMetricSpec(
        metric_dir="ctc_tortoise",
        label="Tortoise CTC Closeness",
        value_field="metric_mean",
        std_field="metric_std",
    ),
    PlotMetricSpec(
        metric_dir="dnsmos",
        label="DNSMOS Overall",
        value_field="metric_mean",
        std_field="metric_std",
    ),
    PlotMetricSpec(
        metric_dir="nisqa",
        label="NISQA MOS",
        value_field="metric_mean",
        std_field="metric_std",
    ),
    PlotMetricSpec(
        metric_dir="speaker_sim",
        label="Speaker Similarity",
        value_field="metric_mean",
        std_field="metric_std",
    ),
    PlotMetricSpec(
        metric_dir="utmos",
        label="UTMOS",
        value_field="metric_mean",
        std_field="metric_std",
    ),
    PlotMetricSpec(
        metric_dir="audiobox",
        label="Audiobox CE",
        value_field="ce_mean",
        std_field="ce_std",
    ),
    PlotMetricSpec(
        metric_dir="audiobox",
        label="Audiobox PQ",
        value_field="pq_mean",
        std_field="pq_std",
    ),
)

FIGURE_FACE_COLOR = "#f7f4ee"
AXES_FACE_COLOR = "#fbfaf7"
GRID_COLOR = "#ddd5c7"
TEXT_COLOR = "#2f2a24"
SPINE_COLOR = "#b7ae9f"
ERROR_BAR_COLOR = "#6c635a"
MODEL_PALETTE = (
    "#355c7d",
    "#c06c84",
    "#6c9a8b",
    "#f2a65a",
    "#7b6d8d",
    "#cc6b5a",
    "#3f7d6b",
    "#d38c5f",
    "#5a6f7b",
    "#b56576",
    "#8f6f98",
    "#4c8c72",
)


def build_plot_series(latest_summaries: dict[str, dict[str, dict[str, Any]]]) -> tuple[list[str], list[PlotSeries]]:
    models = sorted({model for metric_models in latest_summaries.values() for model in metric_models})
    series: list[PlotSeries] = []

    for spec in PLOT_METRICS:
        metric_models = latest_summaries.get(spec.metric_dir, {})
        values_by_model: dict[str, float] = {}
        std_by_model: dict[str, float] = {}

        for model, summary in metric_models.items():
            value = summary.get(spec.value_field)
            if isinstance(value, (int, float)):
                values_by_model[model] = float(value)

            if spec.std_field is None:
                continue

            std_value = summary.get(spec.std_field)
            if isinstance(std_value, (int, float)):
                std_by_model[model] = float(std_value)

        series.append(
            PlotSeries(
                metric_dir=spec.metric_dir,
                label=spec.label,
                values_by_model=values_by_model,
                std_by_model=std_by_model,
            )
        )

    return models, series


def build_grouped_plot_data(
    models: list[str],
    series: list[PlotSeries],
    *,
    group_by: PlotGroupBy = "metric",
) -> GroupedPlotData:
    populated_series = [metric_series for metric_series in series if metric_series.values_by_model]

    if group_by == "metric":
        return GroupedPlotData(
            group_labels=[metric_series.label for metric_series in populated_series],
            bar_labels=models,
            values_by_bar=[
                [metric_series.values_by_model.get(model, float("nan")) for metric_series in populated_series]
                for model in models
            ],
            std_by_bar=[
                [metric_series.std_by_model.get(model, 0.0) for metric_series in populated_series]
                for model in models
            ],
            x_axis_label="Eval Metric",
            legend_title="Model",
        )

    if group_by == "model":
        return GroupedPlotData(
            group_labels=models,
            bar_labels=[metric_series.label for metric_series in populated_series],
            values_by_bar=[
                [metric_series.values_by_model.get(model, float("nan")) for model in models]
                for metric_series in populated_series
            ],
            std_by_bar=[
                [metric_series.std_by_model.get(model, 0.0) for model in models]
                for metric_series in populated_series
            ],
            x_axis_label="Model",
            legend_title="Eval Metric",
        )

    raise ValueError(f"unsupported plot grouping: {group_by}")


def slugify_label(label: str) -> str:
    collapsed = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    return collapsed or "metric"


def build_metric_plot_data(models: list[str], series: list[PlotSeries]) -> list[MetricPlotData]:
    metric_plots: list[MetricPlotData] = []

    for metric_series in series:
        available_models = [
            (
                model,
                metric_series.values_by_model[model],
                metric_series.std_by_model.get(model, 0.0),
            )
            for model in models
            if model in metric_series.values_by_model
        ]
        if not available_models:
            continue

        available_models.sort(key=lambda item: item[1], reverse=True)
        metric_plots.append(
            MetricPlotData(
                metric_dir=metric_series.metric_dir,
                label=metric_series.label,
                slug=slugify_label(metric_series.label),
                model_labels=[item[0] for item in available_models],
                values=[item[1] for item in available_models],
                std_values=[item[2] for item in available_models],
            )
        )

    return metric_plots


def _build_model_color_map(models: list[str]) -> dict[str, str]:
    color_map = {model: MODEL_PALETTE[index % len(MODEL_PALETTE)] for index, model in enumerate(models)}

    if len(models) <= len(MODEL_PALETTE):
        return color_map

    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    fallback_colors = [mcolors.to_hex(color) for color in plt.get_cmap("tab20").colors]
    for index, model in enumerate(models):
        if index < len(MODEL_PALETTE):
            continue
        color_map[model] = fallback_colors[(index - len(MODEL_PALETTE)) % len(fallback_colors)]
    return color_map


def _base_output_path(output_path: Path) -> Path:
    if output_path.suffix:
        return output_path.with_suffix("")
    return output_path


def _metric_output_base(output_path: Path, metric_plot: MetricPlotData) -> Path:
    root = _base_output_path(output_path)
    return root.parent / f"{root.name}_metrics" / metric_plot.slug


def _save_figure_formats(fig: Any, output_base: Path, *, dpi: int) -> tuple[Path, Path]:
    png_path = output_base.with_suffix(".png")
    svg_path = output_base.with_suffix(".svg")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(svg_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    return png_path, svg_path


def _style_axes(ax: Any) -> None:
    ax.set_facecolor(AXES_FACE_COLOR)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)


def _format_model_label(label: str) -> str:
    return label.replace("_", " ").replace("-", " ")


def _draw_metric_bars(
    ax: Any,
    metric_plot: MetricPlotData,
    *,
    model_colors: dict[str, str],
    include_stddev: bool,
) -> None:
    x_positions = list(range(len(metric_plot.model_labels)))
    bar_colors = [model_colors[model] for model in metric_plot.model_labels]
    yerr = None
    if include_stddev and any(value > 0 for value in metric_plot.std_values):
        yerr = metric_plot.std_values

    bars = ax.bar(
        x_positions,
        metric_plot.values,
        color=bar_colors,
        edgecolor=SPINE_COLOR,
        linewidth=0.8,
        width=0.72,
        yerr=yerr,
        capsize=3 if yerr is not None else 0,
        error_kw={"elinewidth": 0.9, "ecolor": ERROR_BAR_COLOR, "capthick": 0.9},
    )

    labels = [_format_model_label(model) for model in metric_plot.model_labels]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title(metric_plot.label, fontsize=12, pad=10, fontweight="semibold")
    ax.set_ylabel("Mean Score", fontsize=10)

    value_max = max(metric_plot.values)
    ax.set_ylim(0, value_max * 1.18 if value_max > 0 else 1.0)

    if len(metric_plot.model_labels) <= 8:
        offset = max(value_max * 0.02, 0.01)
        for bar, value in zip(bars, metric_plot.values, strict=True):
            ax.text(
                bar.get_x() + (bar.get_width() / 2.0),
                value + offset,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=TEXT_COLOR,
            )

    _style_axes(ax)


def _render_combined_figure(
    metric_plots: list[MetricPlotData],
    output_path: Path,
    *,
    model_colors: dict[str, str],
    title: str | None,
    include_stddev: bool,
    dpi: int,
) -> tuple[Path, Path]:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    plot_count = len(metric_plots)
    columns = 2 if plot_count <= 4 else 3
    rows = (plot_count + columns - 1) // columns
    fig_width = 7.2 * columns
    fig_height = 4.6 * rows + 0.9
    fig, axes = plt.subplots(rows, columns, figsize=(fig_width, fig_height), squeeze=False)
    fig.patch.set_facecolor(FIGURE_FACE_COLOR)

    for axis, metric_plot in zip(axes.flat, metric_plots, strict=False):
        _draw_metric_bars(
            axis,
            metric_plot,
            model_colors=model_colors,
            include_stddev=include_stddev,
        )

    for axis in axes.flat[len(metric_plots):]:
        axis.set_visible(False)

    legend_handles = [
        mpatches.Patch(color=color, label=_format_model_label(model))
        for model, color in model_colors.items()
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=min(4, max(1, len(legend_handles))),
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        fontsize=9,
        labelcolor=TEXT_COLOR,
    )
    fig.suptitle(title or "Mean Eval Scores by Metric", fontsize=16, fontweight="semibold", color=TEXT_COLOR, y=1.05)
    saved_paths = _save_figure_formats(fig, _base_output_path(output_path), dpi=dpi)
    plt.close(fig)
    return saved_paths


def _render_metric_figures(
    metric_plots: list[MetricPlotData],
    output_path: Path,
    *,
    model_colors: dict[str, str],
    include_stddev: bool,
    dpi: int,
) -> tuple[dict[str, Path], dict[str, Path]]:
    import matplotlib.pyplot as plt

    metric_pngs: dict[str, Path] = {}
    metric_svgs: dict[str, Path] = {}

    for metric_plot in metric_plots:
        fig_width = max(6.5, len(metric_plot.model_labels) * 1.25)
        fig, ax = plt.subplots(figsize=(fig_width, 5.0))
        fig.patch.set_facecolor(FIGURE_FACE_COLOR)
        _draw_metric_bars(
            ax,
            metric_plot,
            model_colors=model_colors,
            include_stddev=include_stddev,
        )
        fig.suptitle(f"{metric_plot.label} by Model", fontsize=15, fontweight="semibold", color=TEXT_COLOR, y=1.02)
        png_path, svg_path = _save_figure_formats(fig, _metric_output_base(output_path, metric_plot), dpi=dpi)
        plt.close(fig)
        metric_pngs[metric_plot.label] = png_path
        metric_svgs[metric_plot.label] = svg_path

    return metric_pngs, metric_svgs


def render_mean_plot_outputs(
    latest_summaries: dict[str, dict[str, dict[str, Any]]],
    output_path: Path,
    title: str | None = None,
    include_stddev: bool = False,
    dpi: int = 180,
) -> RenderedPlotOutputs:
    models, series = build_plot_series(latest_summaries)

    if not models:
        raise ValueError("no model summaries found under eval root")

    metric_plots = build_metric_plot_data(models, series)
    if not metric_plots:
        raise ValueError("no plottable metric values found in latest summaries")

    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"missing plotting dependency: {exc.name}") from exc

    model_colors = _build_model_color_map(models)
    combined_png, combined_svg = _render_combined_figure(
        metric_plots,
        output_path,
        model_colors=model_colors,
        title=title,
        include_stddev=include_stddev,
        dpi=dpi,
    )
    metric_pngs, metric_svgs = _render_metric_figures(
        metric_plots,
        output_path,
        model_colors=model_colors,
        include_stddev=include_stddev,
        dpi=dpi,
    )
    return RenderedPlotOutputs(
        combined_png=combined_png,
        combined_svg=combined_svg,
        metric_pngs=metric_pngs,
        metric_svgs=metric_svgs,
    )


def render_mean_plot(
    latest_summaries: dict[str, dict[str, dict[str, Any]]],
    output_path: Path,
    title: str | None = None,
    include_stddev: bool = False,
    dpi: int = 180,
    group_by: PlotGroupBy = "metric",
) -> Path:
    _ = group_by
    return render_mean_plot_outputs(
        latest_summaries=latest_summaries,
        output_path=output_path,
        title=title,
        include_stddev=include_stddev,
        dpi=dpi,
    ).combined_png
