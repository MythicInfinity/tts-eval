from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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


PLOT_METRICS = (
    PlotMetricSpec(
        metric_dir="ctc",
        label="CTC Closeness",
        value_field="metric_mean",
        std_field="metric_std",
    ),
    PlotMetricSpec(
        metric_dir="ttsds2",
        label="TTSDS2 Total",
        value_field="metric_value",
        std_field=None,
    ),
    PlotMetricSpec(
        metric_dir="dnsmos",
        label="DNSMOS Overall",
        value_field="metric_mean",
        std_field="metric_std",
    ),
    PlotMetricSpec(
        metric_dir="speaker_sim",
        label="Speaker Similarity",
        value_field="metric_mean",
        std_field="metric_std",
    ),
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


def render_mean_plot(
    latest_summaries: dict[str, dict[str, dict[str, Any]]],
    output_path: Path,
    title: str | None = None,
    include_stddev: bool = False,
    dpi: int = 180,
    group_by: PlotGroupBy = "metric",
) -> Path:
    models, series = build_plot_series(latest_summaries)

    if not models:
        raise ValueError("no model summaries found under eval root")

    plot_data = build_grouped_plot_data(models, series, group_by=group_by)
    if not plot_data.group_labels or not plot_data.bar_labels:
        raise ValueError("no plottable metric values found in latest summaries")

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"missing plotting dependency: {exc.name}") from exc

    x = np.arange(len(plot_data.group_labels))
    width = 0.8 / len(plot_data.bar_labels)
    offsets = np.linspace(-0.4 + (width / 2.0), 0.4 - (width / 2.0), len(plot_data.bar_labels))

    fig, ax = plt.subplots(figsize=(max(10, len(plot_data.group_labels) * 1.8), 6))

    for offset, bar_label, heights, std_values in zip(
        offsets,
        plot_data.bar_labels,
        plot_data.values_by_bar,
        plot_data.std_by_bar,
        strict=True,
    ):
        yerr = None
        if include_stddev:
            if any(value > 0 for value in std_values):
                yerr = std_values

        ax.bar(
            x + offset,
            heights,
            width=width,
            label=bar_label,
            yerr=yerr,
            capsize=4 if yerr is not None else 0,
            alpha=0.9,
        )

    ax.set_title(title or f"Mean Eval Scores By {'Metric' if group_by == 'metric' else 'Model'}")
    ax.set_xlabel(plot_data.x_axis_label)
    ax.set_ylabel("Mean Score")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_data.group_labels, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title=plot_data.legend_title)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path
