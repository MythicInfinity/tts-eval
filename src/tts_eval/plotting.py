from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


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


def render_mean_plot(
    latest_summaries: dict[str, dict[str, dict[str, Any]]],
    output_path: Path,
    title: str = "Mean Eval Scores By Model",
    include_stddev: bool = False,
    dpi: int = 180,
) -> Path:
    models, series = build_plot_series(latest_summaries)

    if not models:
        raise ValueError("no model summaries found under eval root")

    populated_series = [metric_series for metric_series in series if metric_series.values_by_model]
    if not populated_series:
        raise ValueError("no plottable metric values found in latest summaries")

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"missing plotting dependency: {exc.name}") from exc

    x = np.arange(len(models))
    width = 0.8 / len(populated_series)
    offsets = np.linspace(-0.4 + (width / 2.0), 0.4 - (width / 2.0), len(populated_series))

    fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.4), 6))

    for offset, metric_series in zip(offsets, populated_series, strict=True):
        heights = [metric_series.values_by_model.get(model, float("nan")) for model in models]
        yerr = None
        if include_stddev:
            std_values = [metric_series.std_by_model.get(model, 0.0) for model in models]
            if any(value > 0 for value in std_values):
                yerr = std_values

        ax.bar(
            x + offset,
            heights,
            width=width,
            label=metric_series.label,
            yerr=yerr,
            capsize=4 if yerr is not None else 0,
            alpha=0.9,
        )

    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean Score")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path
