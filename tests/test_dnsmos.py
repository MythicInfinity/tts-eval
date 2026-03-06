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
    extract_overall_dnsmos,
)


class DNSMOSHelpersTests(unittest.TestCase):
    def test_extract_overall_dnsmos_from_nested_list(self) -> None:
        self.assertEqual(extract_overall_dnsmos([[1.0, 2.0, 3.0, 4.0]]), 4.0)

    def test_extract_overall_dnsmos_rejects_invalid_shape(self) -> None:
        with self.assertRaises(RuntimeError):
            extract_overall_dnsmos([1.0, 2.0, 3.0])


class DNSMOSEvaluationTests(unittest.TestCase):
    def test_evaluate_model_accumulates_meaningful_audio_duration(self) -> None:
        runtime = SimpleNamespace(metric_version="v1")

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            wav_path = model_dir / "speaker01_00001.wav"
            wav_path.write_bytes(b"wav")

            with mock.patch("tts_eval.dnsmos.load_audio_sample", return_value=AudioSample(waveform=None, sample_rate=16000, duration_sec=1.5)):
                with mock.patch("tts_eval.dnsmos.score_audio_sample", return_value=3.4):
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

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            wav_path = model_dir / "speaker01_00001.wav"
            wav_path.write_bytes(b"wav")

            with mock.patch("tts_eval.dnsmos.load_audio_sample", return_value=AudioSample(waveform=None, sample_rate=16000, duration_sec=1.0)):
                with mock.patch("tts_eval.dnsmos.score_audio_sample", side_effect=RuntimeError("backend exploded")):
                    records, total_audio_sec, _ = evaluate_model("model_a", model_dir, runtime, "2026-03-06T00:00:00Z")

        self.assertEqual(total_audio_sec, 1.0)
        self.assertEqual(records[0].status, "fail")
        self.assertEqual(records[0].error, "backend exploded")


if __name__ == "__main__":
    unittest.main()
