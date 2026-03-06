from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tts_eval.dnsmos import (
    AudioSample,
    SkipUtteranceError,
    evaluate_model,
    extract_batch_overall_dnsmos,
    extract_overall_dnsmos,
)


class DNSMOSHelpersTests(unittest.TestCase):
    def test_extract_overall_dnsmos_from_nested_list(self) -> None:
        self.assertEqual(extract_overall_dnsmos([[1.0, 2.0, 3.0, 4.0]]), 4.0)

    def test_extract_overall_dnsmos_rejects_invalid_shape(self) -> None:
        with self.assertRaises(RuntimeError):
            extract_overall_dnsmos([1.0, 2.0, 3.0])

    def test_extract_batch_overall_dnsmos(self) -> None:
        self.assertEqual(
            extract_batch_overall_dnsmos(
                [[1.0, 2.0, 3.0, 4.0], [1.2, 2.2, 3.2, 4.2]],
                expected_count=2,
            ),
            [4.0, 4.2],
        )


class DNSMOSEvaluationTests(unittest.TestCase):
    def test_evaluate_model_accumulates_meaningful_audio_duration(self) -> None:
        runtime = SimpleNamespace(metric_version="v1")
        waveform = SimpleNamespace(shape=(1, 16000))

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            wav_path = model_dir / "speaker01_00001.wav"
            wav_path.write_bytes(b"wav")

            with mock.patch("tts_eval.dnsmos.load_audio_sample", return_value=AudioSample(waveform=waveform, sample_rate=16000, duration_sec=1.5)):
                with mock.patch("tts_eval.dnsmos.score_audio_batch", return_value=[3.4]):
                    records, total_audio_sec, n_utts = evaluate_model("model_a", model_dir, runtime, "2026-03-06T00:00:00Z")

        self.assertEqual(n_utts, 1)
        self.assertEqual(total_audio_sec, 1.5)
        self.assertEqual(records[0].status, "ok")
        self.assertEqual(records[0].metric_value, 3.4)

    def test_unreadable_wav_is_skipped(self) -> None:
        runtime = SimpleNamespace(metric_version="v1")

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            wav_path = model_dir / "speaker01_00001.wav"
            wav_path.write_bytes(b"wav")

            with mock.patch("tts_eval.dnsmos.load_audio_sample", side_effect=SkipUtteranceError("unreadable wav: bad header")):
                records, total_audio_sec, n_utts = evaluate_model("model_a", model_dir, runtime, "2026-03-06T00:00:00Z")

        self.assertEqual(n_utts, 1)
        self.assertEqual(total_audio_sec, 0.0)
        self.assertEqual(records[0].status, "skip")
        self.assertEqual(records[0].error, "unreadable wav: bad header")

    def test_backend_failure_is_reported_as_fail(self) -> None:
        runtime = SimpleNamespace(metric_version="v1")
        waveform = SimpleNamespace(shape=(1, 16000))

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            wav_path = model_dir / "speaker01_00001.wav"
            wav_path.write_bytes(b"wav")

            with mock.patch("tts_eval.dnsmos.load_audio_sample", return_value=AudioSample(waveform=waveform, sample_rate=16000, duration_sec=1.0)):
                with mock.patch("tts_eval.dnsmos.score_audio_batch", side_effect=RuntimeError("batch exploded")):
                    with mock.patch("tts_eval.dnsmos.score_audio_sample", side_effect=RuntimeError("backend exploded")):
                        records, total_audio_sec, _ = evaluate_model("model_a", model_dir, runtime, "2026-03-06T00:00:00Z")

        self.assertEqual(total_audio_sec, 1.0)
        self.assertEqual(records[0].status, "fail")
        self.assertEqual(records[0].error, "backend exploded")

    def test_evaluate_model_batches_multiple_utterances(self) -> None:
        runtime = SimpleNamespace(metric_version="v1")
        waveform_a = SimpleNamespace(shape=(1, 16000))
        waveform_b = SimpleNamespace(shape=(1, 16000))

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "speaker01_00001.wav").write_bytes(b"wav")
            (model_dir / "speaker01_00002.wav").write_bytes(b"wav")

            load_side_effect = [
                AudioSample(waveform=waveform_a, sample_rate=16000, duration_sec=1.0),
                AudioSample(waveform=waveform_b, sample_rate=16000, duration_sec=2.0),
            ]

            with mock.patch("tts_eval.dnsmos.load_audio_sample", side_effect=load_side_effect):
                with mock.patch("tts_eval.dnsmos.score_audio_batch", return_value=[3.0, 3.5]) as batch_mock:
                    with mock.patch("tts_eval.dnsmos.score_audio_sample") as single_mock:
                        records, total_audio_sec, n_utts = evaluate_model(
                            "model_a",
                            model_dir,
                            runtime,
                            "2026-03-06T00:00:00Z",
                            batch_size=8,
                        )

        self.assertEqual(n_utts, 2)
        self.assertEqual(total_audio_sec, 3.0)
        self.assertEqual([record.status for record in records], ["ok", "ok"])
        self.assertEqual([record.metric_value for record in records], [3.0, 3.5])
        self.assertEqual(batch_mock.call_count, 1)
        self.assertEqual(single_mock.call_count, 0)

    def test_evaluate_model_splits_batch_on_sample_rate_change(self) -> None:
        runtime = SimpleNamespace(metric_version="v1")
        waveform_a = SimpleNamespace(shape=(1, 16000))
        waveform_b = SimpleNamespace(shape=(1, 16000))

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "speaker01_00001.wav").write_bytes(b"wav")
            (model_dir / "speaker01_00002.wav").write_bytes(b"wav")

            load_side_effect = [
                AudioSample(waveform=waveform_a, sample_rate=16000, duration_sec=1.0),
                AudioSample(waveform=waveform_b, sample_rate=24000, duration_sec=1.0),
            ]

            with mock.patch("tts_eval.dnsmos.load_audio_sample", side_effect=load_side_effect):
                with mock.patch("tts_eval.dnsmos.score_audio_batch", side_effect=[[3.1], [3.2]]) as batch_mock:
                    records, total_audio_sec, _ = evaluate_model("model_a", model_dir, runtime, "2026-03-06T00:00:00Z")

        self.assertEqual(total_audio_sec, 2.0)
        self.assertEqual([record.metric_value for record in records], [3.1, 3.2])
        self.assertEqual(batch_mock.call_count, 2)

    def test_evaluate_model_splits_batch_on_length_change(self) -> None:
        runtime = SimpleNamespace(metric_version="v1")
        waveform_a = SimpleNamespace(shape=(1, 16000))
        waveform_b = SimpleNamespace(shape=(1, 20000))

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "speaker01_00001.wav").write_bytes(b"wav")
            (model_dir / "speaker01_00002.wav").write_bytes(b"wav")

            load_side_effect = [
                AudioSample(waveform=waveform_a, sample_rate=16000, duration_sec=1.0),
                AudioSample(waveform=waveform_b, sample_rate=16000, duration_sec=1.2),
            ]

            with mock.patch("tts_eval.dnsmos.load_audio_sample", side_effect=load_side_effect):
                with mock.patch("tts_eval.dnsmos.score_audio_batch", side_effect=[[3.1], [3.2]]) as batch_mock:
                    records, total_audio_sec, _ = evaluate_model("model_a", model_dir, runtime, "2026-03-06T00:00:00Z")

        self.assertEqual(total_audio_sec, 2.2)
        self.assertEqual([record.metric_value for record in records], [3.1, 3.2])
        self.assertEqual(batch_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
