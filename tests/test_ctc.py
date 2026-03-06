from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tts_eval.ctc import (
    AudioSample,
    SkipUtteranceError,
    ctc_closeness_from_loss,
    decode_greedy,
    evaluate_model,
    tokenize_transcript,
)
from tts_eval.io import MetricRecord
from tts_eval.stats import aggregate_metric_records


class TranscriptTokenizationTests(unittest.TestCase):
    def test_tokenize_transcript_keeps_supported_labels(self) -> None:
        labels = ("-", "A", "B", "'", "|")
        tokenized = tokenize_transcript("ab a!'?", labels)
        self.assertEqual(tokenized.normalized_text, "AB|A'")
        self.assertEqual(tokenized.token_ids, (1, 2, 4, 1, 3))

    def test_tokenize_transcript_skips_when_empty_after_filtering(self) -> None:
        labels = ("-", "A", "B")
        with self.assertRaises(SkipUtteranceError):
            tokenize_transcript("!? 123", labels)

    def test_tokenize_transcript_drops_blank_symbol_from_targets(self) -> None:
        labels = ("-", "A", "B", "'", "|")
        tokenized = tokenize_transcript("A-B", labels)
        self.assertEqual(tokenized.normalized_text, "AB")
        self.assertEqual(tokenized.token_ids, (1, 2))


class CTCMetricTests(unittest.TestCase):
    def test_ctc_closeness_formula(self) -> None:
        closeness, normalized_loss = ctc_closeness_from_loss(total_loss=6.0, target_length=3)
        self.assertEqual(normalized_loss, 2.0)
        self.assertTrue(math.isclose(closeness, math.exp(-2.0)))

    def test_decode_greedy_collapses_repeats_and_blanks(self) -> None:
        labels = ("-", "A", "B", "|")
        decoded = decode_greedy([0, 1, 1, 0, 3, 2, 2, 0], labels)
        self.assertEqual(decoded, "A B")


class AggregateStatsTests(unittest.TestCase):
    def test_aggregate_metric_records_ignores_non_ok_values(self) -> None:
        records = [
            MetricRecord(
                run_timestamp_utc="2026-03-06T00:00:00Z",
                metric_name="ctc_closeness",
                metric_version="v1",
                model="model_a",
                utt_id="speaker01_00001",
                wav_path="/tmp/a.wav",
                metric_value=0.9,
                status="ok",
                error=None,
            ),
            MetricRecord(
                run_timestamp_utc="2026-03-06T00:00:00Z",
                metric_name="ctc_closeness",
                metric_version="v1",
                model="model_a",
                utt_id="speaker01_00002",
                wav_path="/tmp/b.wav",
                metric_value=None,
                status="skip",
                error="missing transcript sidecar",
            ),
            MetricRecord(
                run_timestamp_utc="2026-03-06T00:00:00Z",
                metric_name="ctc_closeness",
                metric_version="v1",
                model="model_a",
                utt_id="speaker01_00003",
                wav_path="/tmp/c.wav",
                metric_value=None,
                status="fail",
                error="boom",
            ),
        ]
        aggregate = aggregate_metric_records(records)
        self.assertEqual(aggregate.metric_mean, 0.9)
        self.assertEqual(aggregate.metric_median, 0.9)
        self.assertEqual(aggregate.metric_std, 0.0)
        self.assertEqual(aggregate.skip_count, 1)
        self.assertEqual(aggregate.fail_count, 1)

    def test_aggregate_metric_records_returns_null_stats_without_successes(self) -> None:
        records = [
            MetricRecord(
                run_timestamp_utc="2026-03-06T00:00:00Z",
                metric_name="ctc_closeness",
                metric_version="v1",
                model="model_a",
                utt_id="speaker01_00001",
                wav_path="/tmp/a.wav",
                metric_value=None,
                status="skip",
                error="missing transcript sidecar",
            )
        ]
        aggregate = aggregate_metric_records(records)
        self.assertIsNone(aggregate.metric_mean)
        self.assertIsNone(aggregate.metric_median)
        self.assertIsNone(aggregate.metric_std)


class EvaluateModelTests(unittest.TestCase):
    def test_missing_transcript_still_counts_audio_duration(self) -> None:
        runtime = SimpleNamespace(metric_version="v1")

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            wav_path = model_dir / "speaker01_00001.wav"
            wav_path.write_bytes(b"not-a-real-wav")

            with mock.patch("tts_eval.ctc.load_audio_sample", return_value=AudioSample(waveform=None, sample_rate=16000, duration_sec=1.25)):
                records, total_audio_sec, n_utts = evaluate_model("model_a", model_dir, runtime, "2026-03-06T00:00:00Z")

        self.assertEqual(n_utts, 1)
        self.assertEqual(total_audio_sec, 1.25)
        self.assertEqual(records[0].status, "skip")
        self.assertEqual(records[0].error, "missing transcript sidecar")

    def test_unreadable_wav_is_skipped(self) -> None:
        runtime = SimpleNamespace(metric_version="v1")

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            wav_path = model_dir / "speaker01_00001.wav"
            txt_path = model_dir / "speaker01_00001.txt"
            wav_path.write_bytes(b"not-a-real-wav")
            txt_path.write_text("hello", encoding="utf-8")

            with mock.patch("tts_eval.ctc.load_audio_sample", side_effect=SkipUtteranceError("unreadable wav: bad header")):
                records, total_audio_sec, n_utts = evaluate_model("model_a", model_dir, runtime, "2026-03-06T00:00:00Z")

        self.assertEqual(n_utts, 1)
        self.assertEqual(total_audio_sec, 0.0)
        self.assertEqual(records[0].status, "skip")
        self.assertEqual(records[0].error, "unreadable wav: bad header")


if __name__ == "__main__":
    unittest.main()
