from __future__ import annotations

import unittest

from tts_eval.plotting import build_plot_series


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
            }
        )

        self.assertEqual(models, ["model_a", "model_b"])
        self.assertEqual(series[0].values_by_model, {"model_a": 0.91, "model_b": 0.88})
        self.assertEqual(series[0].std_by_model, {"model_a": 0.03, "model_b": 0.02})
        self.assertEqual(series[1].values_by_model, {"model_a": 0.77, "model_b": 0.79})
        self.assertEqual(series[1].std_by_model, {})
        self.assertEqual(series[2].values_by_model, {"model_a": 3.4})
        self.assertEqual(series[2].std_by_model, {"model_a": 0.1})

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


if __name__ == "__main__":
    unittest.main()
