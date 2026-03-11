from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

from tts_eval.plotting import (
    build_grouped_plot_data,
    build_metric_plot_data,
    build_plot_series,
    render_mean_plot_outputs,
)


SAMPLE_SUMMARIES = {
    "ctc": {
        "model_a": {"metric_mean": 0.91, "metric_std": 0.03},
        "model_b": {"metric_mean": 0.88, "metric_std": 0.02},
    },
    "ctc_tortoise": {
        "model_a": {"metric_mean": 0.89, "metric_std": 0.02},
        "model_b": {"metric_mean": 0.84, "metric_std": 0.01},
    },
    "dnsmos": {
        "model_a": {"metric_mean": 3.4, "metric_std": 0.1},
    },
    "nisqa": {
        "model_a": {"metric_mean": 3.9, "metric_std": 0.2},
        "model_b": {"metric_mean": 3.7, "metric_std": 0.1},
    },
    "speaker_sim": {
        "model_a": {"metric_mean": 0.82, "metric_std": 0.04},
        "model_b": {"metric_mean": 0.78, "metric_std": 0.03},
    },
    "utmos": {
        "model_a": {"metric_mean": 4.1, "metric_std": 0.12},
        "model_b": {"metric_mean": 4.0, "metric_std": 0.08},
    },
    "audiobox": {
        "model_a": {"ce_mean": 5.2, "ce_std": 0.3, "pq_mean": 6.1, "pq_std": 0.4},
        "model_b": {"ce_mean": 5.0, "ce_std": 0.2, "pq_mean": 5.8, "pq_std": 0.3},
    },
}


class PlottingTests(unittest.TestCase):
    def test_build_plot_series_uses_means(self) -> None:
        models, series = build_plot_series(SAMPLE_SUMMARIES)

        self.assertEqual(models, ["model_a", "model_b"])
        self.assertEqual([item.label for item in series], [
            "CTC Closeness",
            "Tortoise CTC Closeness",
            "DNSMOS Overall",
            "NISQA MOS",
            "Speaker Similarity",
            "UTMOS",
            "Audiobox CE",
            "Audiobox PQ",
        ])
        by_label = {item.label: item for item in series}

        self.assertEqual(by_label["CTC Closeness"].values_by_model, {"model_a": 0.91, "model_b": 0.88})
        self.assertEqual(by_label["Tortoise CTC Closeness"].values_by_model, {"model_a": 0.89, "model_b": 0.84})
        self.assertEqual(by_label["DNSMOS Overall"].values_by_model, {"model_a": 3.4})
        self.assertEqual(by_label["NISQA MOS"].values_by_model, {"model_a": 3.9, "model_b": 3.7})
        self.assertEqual(by_label["Speaker Similarity"].values_by_model, {"model_a": 0.82, "model_b": 0.78})
        self.assertEqual(by_label["UTMOS"].values_by_model, {"model_a": 4.1, "model_b": 4.0})
        self.assertEqual(by_label["Audiobox CE"].values_by_model, {"model_a": 5.2, "model_b": 5.0})
        self.assertEqual(by_label["Audiobox PQ"].values_by_model, {"model_a": 6.1, "model_b": 5.8})

    def test_build_plot_series_ignores_non_numeric_values(self) -> None:
        models, series = build_plot_series(
            {
                "ctc": {
                    "model_a": {"metric_mean": None, "metric_std": 0.03},
                },
                "ctc_tortoise": {
                    "model_a": {"metric_mean": None, "metric_std": 0.02},
                },
                "dnsmos": {
                    "model_a": {"metric_mean": "bad", "metric_std": 0.1},
                },
                "nisqa": {
                    "model_a": {"metric_mean": None, "metric_std": 0.2},
                },
                "speaker_sim": {
                    "model_a": {"metric_mean": None, "metric_std": 0.04},
                },
                "utmos": {
                    "model_a": {"metric_mean": "bad", "metric_std": 0.12},
                },
                "audiobox": {
                    "model_a": {"ce_mean": None, "ce_std": 0.3, "pq_mean": "bad", "pq_std": 0.4},
                },
            }
        )

        self.assertEqual(models, ["model_a"])
        for item in series:
            self.assertEqual(item.values_by_model, {})

    def test_build_grouped_plot_data_groups_by_metric_by_default(self) -> None:
        models, series = build_plot_series(SAMPLE_SUMMARIES)

        grouped = build_grouped_plot_data(models, series)

        self.assertEqual(
            grouped.group_labels,
            [
                "CTC Closeness",
                "Tortoise CTC Closeness",
                "DNSMOS Overall",
                "NISQA MOS",
                "Speaker Similarity",
                "UTMOS",
                "Audiobox CE",
                "Audiobox PQ",
            ],
        )
        self.assertEqual(grouped.bar_labels, ["model_a", "model_b"])
        self.assertEqual(grouped.values_by_bar[0], [0.91, 0.89, 3.4, 3.9, 0.82, 4.1, 5.2, 6.1])
        self.assertEqual(grouped.std_by_bar[0], [0.03, 0.02, 0.1, 0.2, 0.04, 0.12, 0.3, 0.4])
        self.assertEqual(grouped.values_by_bar[1][0], 0.88)
        self.assertEqual(grouped.values_by_bar[1][1], 0.84)
        self.assertTrue(math.isnan(grouped.values_by_bar[1][2]))
        self.assertEqual(grouped.values_by_bar[1][3:], [3.7, 0.78, 4.0, 5.0, 5.8])
        self.assertEqual(grouped.std_by_bar[1], [0.02, 0.01, 0.0, 0.1, 0.03, 0.08, 0.2, 0.3])
        self.assertEqual(grouped.x_axis_label, "Eval Metric")
        self.assertEqual(grouped.legend_title, "Model")

    def test_build_grouped_plot_data_can_group_by_model(self) -> None:
        models, series = build_plot_series(SAMPLE_SUMMARIES)

        grouped = build_grouped_plot_data(models, series, group_by="model")

        self.assertEqual(grouped.group_labels, ["model_a", "model_b"])
        self.assertEqual(
            grouped.bar_labels,
            [
                "CTC Closeness",
                "Tortoise CTC Closeness",
                "DNSMOS Overall",
                "NISQA MOS",
                "Speaker Similarity",
                "UTMOS",
                "Audiobox CE",
                "Audiobox PQ",
            ],
        )
        self.assertEqual(grouped.values_by_bar[0], [0.91, 0.88])
        self.assertEqual(grouped.std_by_bar[0], [0.03, 0.02])
        self.assertEqual(grouped.values_by_bar[1], [0.89, 0.84])
        self.assertEqual(grouped.std_by_bar[1], [0.02, 0.01])
        self.assertEqual(grouped.values_by_bar[2][0], 3.4)
        self.assertTrue(math.isnan(grouped.values_by_bar[2][1]))
        self.assertEqual(grouped.std_by_bar[2], [0.1, 0.0])
        self.assertEqual(grouped.values_by_bar[3], [3.9, 3.7])
        self.assertEqual(grouped.std_by_bar[3], [0.2, 0.1])
        self.assertEqual(grouped.values_by_bar[4], [0.82, 0.78])
        self.assertEqual(grouped.std_by_bar[4], [0.04, 0.03])
        self.assertEqual(grouped.values_by_bar[5], [4.1, 4.0])
        self.assertEqual(grouped.std_by_bar[5], [0.12, 0.08])
        self.assertEqual(grouped.values_by_bar[6], [5.2, 5.0])
        self.assertEqual(grouped.std_by_bar[6], [0.3, 0.2])
        self.assertEqual(grouped.values_by_bar[7], [6.1, 5.8])
        self.assertEqual(grouped.std_by_bar[7], [0.4, 0.3])
        self.assertEqual(grouped.x_axis_label, "Model")
        self.assertEqual(grouped.legend_title, "Eval Metric")

    def test_build_metric_plot_data_sorts_models_within_each_metric(self) -> None:
        models, series = build_plot_series(SAMPLE_SUMMARIES)

        metric_plots = build_metric_plot_data(models, series)
        by_label = {item.label: item for item in metric_plots}

        self.assertEqual(by_label["CTC Closeness"].model_labels, ["model_a", "model_b"])
        self.assertEqual(by_label["DNSMOS Overall"].model_labels, ["model_a"])
        self.assertEqual(by_label["Audiobox PQ"].values, [6.1, 5.8])
        self.assertEqual(by_label["Audiobox PQ"].slug, "audiobox-pq")

    def test_render_mean_plot_outputs_creates_combined_and_metric_files(self) -> None:
        try:
            import matplotlib  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("matplotlib is not installed")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "mean_eval_plot.png"
            outputs = render_mean_plot_outputs(
                SAMPLE_SUMMARIES,
                output_path=output_path,
                title="Styled Eval Plots",
                include_stddev=True,
                dpi=100,
            )

            self.assertTrue(outputs.combined_png.exists())
            self.assertTrue(outputs.combined_svg.exists())
            self.assertEqual(outputs.combined_png.name, "mean_eval_plot.png")
            self.assertEqual(outputs.combined_svg.name, "mean_eval_plot.svg")

            self.assertIn("CTC Closeness", outputs.metric_pngs)
            self.assertIn("CTC Closeness", outputs.metric_svgs)
            self.assertTrue(outputs.metric_pngs["CTC Closeness"].exists())
            self.assertTrue(outputs.metric_svgs["CTC Closeness"].exists())
            self.assertEqual(outputs.metric_pngs["CTC Closeness"].parent.name, "mean_eval_plot_metrics")


if __name__ == "__main__":
    unittest.main()
