from __future__ import annotations

import math
import unittest

from tts_eval.plotting import build_grouped_plot_data, build_plot_series


class PlottingTests(unittest.TestCase):
    def test_build_plot_series_uses_means_and_metric_values(self) -> None:
        models, series = build_plot_series(
            {
                "ctc": {
                    "model_a": {"metric_mean": 0.91, "metric_std": 0.03},
                    "model_b": {"metric_mean": 0.88, "metric_std": 0.02},
                },
                "ttsds2": {
                    "model_a": {"metric_value": 0.77},
                    "model_b": {"metric_value": 0.79},
                },
                "dnsmos": {
                    "model_a": {"metric_mean": 3.4, "metric_std": 0.1},
                },
                "nisqa": {
                    "model_b": {"metric_mean": 4.2, "metric_std": 0.05},
                },
            }
        )

        self.assertEqual(models, ["model_a", "model_b"])
        self.assertEqual(series[0].values_by_model, {"model_a": 0.91, "model_b": 0.88})
        self.assertEqual(series[0].std_by_model, {"model_a": 0.03, "model_b": 0.02})
        self.assertEqual(series[1].values_by_model, {"model_a": 0.77, "model_b": 0.79})
        self.assertEqual(series[1].std_by_model, {})
        self.assertEqual(series[2].values_by_model, {"model_a": 3.4})
        self.assertEqual(series[2].std_by_model, {"model_a": 0.1})
        self.assertEqual(series[3].values_by_model, {"model_b": 4.2})
        self.assertEqual(series[3].std_by_model, {"model_b": 0.05})

    def test_build_plot_series_ignores_non_numeric_values(self) -> None:
        models, series = build_plot_series(
            {
                "ctc": {
                    "model_a": {"metric_mean": None, "metric_std": 0.03},
                },
                "ttsds2": {
                    "model_a": {"metric_value": "bad"},
                },
            }
        )

        self.assertEqual(models, ["model_a"])
        self.assertEqual(series[0].values_by_model, {})
        self.assertEqual(series[1].values_by_model, {})

    def test_build_grouped_plot_data_groups_by_metric_by_default(self) -> None:
        models, series = build_plot_series(
            {
                "ctc": {
                    "model_a": {"metric_mean": 0.91, "metric_std": 0.03},
                    "model_b": {"metric_mean": 0.88, "metric_std": 0.02},
                },
                "ttsds2": {
                    "model_a": {"metric_value": 0.77},
                    "model_b": {"metric_value": 0.79},
                },
                "dnsmos": {
                    "model_a": {"metric_mean": 3.4, "metric_std": 0.1},
                },
                "nisqa": {
                    "model_b": {"metric_mean": 4.2, "metric_std": 0.05},
                },
            }
        )

        grouped = build_grouped_plot_data(models, series)

        self.assertEqual(
            grouped.group_labels,
            ["CTC Closeness", "TTSDS2 Total", "DNSMOS Overall", "NISQA MOS"],
        )
        self.assertEqual(grouped.bar_labels, ["model_a", "model_b"])
        self.assertEqual(grouped.values_by_bar[0][:3], [0.91, 0.77, 3.4])
        self.assertTrue(math.isnan(grouped.values_by_bar[0][3]))
        self.assertEqual(grouped.std_by_bar[0], [0.03, 0.0, 0.1, 0.0])
        self.assertEqual(grouped.values_by_bar[1][:2], [0.88, 0.79])
        self.assertTrue(math.isnan(grouped.values_by_bar[1][2]))
        self.assertEqual(grouped.values_by_bar[1][3], 4.2)
        self.assertEqual(grouped.std_by_bar[1], [0.02, 0.0, 0.0, 0.05])
        self.assertEqual(grouped.x_axis_label, "Eval Metric")
        self.assertEqual(grouped.legend_title, "Model")

    def test_build_grouped_plot_data_can_group_by_model(self) -> None:
        models, series = build_plot_series(
            {
                "ctc": {
                    "model_a": {"metric_mean": 0.91, "metric_std": 0.03},
                    "model_b": {"metric_mean": 0.88, "metric_std": 0.02},
                },
                "ttsds2": {
                    "model_a": {"metric_value": 0.77},
                    "model_b": {"metric_value": 0.79},
                },
                "dnsmos": {
                    "model_a": {"metric_mean": 3.4, "metric_std": 0.1},
                },
                "nisqa": {
                    "model_b": {"metric_mean": 4.2, "metric_std": 0.05},
                },
            }
        )

        grouped = build_grouped_plot_data(models, series, group_by="model")

        self.assertEqual(grouped.group_labels, ["model_a", "model_b"])
        self.assertEqual(
            grouped.bar_labels,
            ["CTC Closeness", "TTSDS2 Total", "DNSMOS Overall", "NISQA MOS"],
        )
        self.assertEqual(grouped.values_by_bar[0], [0.91, 0.88])
        self.assertEqual(grouped.std_by_bar[0], [0.03, 0.02])
        self.assertEqual(grouped.values_by_bar[1], [0.77, 0.79])
        self.assertEqual(grouped.std_by_bar[1], [0.0, 0.0])
        self.assertEqual(grouped.values_by_bar[2][0], 3.4)
        self.assertTrue(math.isnan(grouped.values_by_bar[2][1]))
        self.assertEqual(grouped.std_by_bar[2], [0.1, 0.0])
        self.assertTrue(math.isnan(grouped.values_by_bar[3][0]))
        self.assertEqual(grouped.values_by_bar[3][1], 4.2)
        self.assertEqual(grouped.std_by_bar[3], [0.0, 0.05])
        self.assertEqual(grouped.x_axis_label, "Model")
        self.assertEqual(grouped.legend_title, "Eval Metric")


if __name__ == "__main__":
    unittest.main()
