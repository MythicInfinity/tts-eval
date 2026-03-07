from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

from tts_eval.audiobox import (
    AudioSample,
    SkipUtteranceError,
    evaluate_model,
    extract_batch_scores,
    load_audiobox_runtime,
    load_audio_sample,
    score_wav_batch_resilient,
)


class AudioboxHelpersTests(unittest.TestCase):
    def test_extract_batch_scores_reads_ce_and_pq(self) -> None:
        self.assertEqual(
            extract_batch_scores(
                [{"CE": 5.1, "PQ": 6.2}, {"CE": 5.0, "PQ": 6.1}],
                expected_count=2,
            ),
            [(5.1, 6.2), (5.0, 6.1)],
        )

    def test_extract_batch_scores_rejects_invalid_shape(self) -> None:
        with self.assertRaises(RuntimeError):
            extract_batch_scores([{"CE": 5.1, "PQ": 6.2}], expected_count=2)

    def test_extract_batch_scores_rejects_missing_axis(self) -> None:
        with self.assertRaises(RuntimeError):
            extract_batch_scores([{"CE": 5.1}], expected_count=1)


class AudioboxEvaluationTests(unittest.TestCase):
    def test_evaluate_model_accumulates_meaningful_audio_duration(self) -> None:
        runtime = SimpleNamespace(metric_version="v1")
        waveform = SimpleNamespace(shape=(1, 16000))

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            wav_path = model_dir / "speaker01_00001.wav"
            wav_path.write_bytes(b"wav")

            with mock.patch(
                "tts_eval.audiobox.load_audio_sample",
                return_value=AudioSample(waveform=waveform, sample_rate=16000, duration_sec=1.5),
            ):
                with mock.patch(
                    "tts_eval.audiobox.score_wav_batch_resilient",
                    return_value=({str(wav_path.resolve()): (5.2, 6.3)}, {}),
                ):
                    records, total_audio_sec, n_utts = evaluate_model("model_a", model_dir, runtime, "2026-03-06T00:00:00Z")

        self.assertEqual(n_utts, 1)
        self.assertEqual(total_audio_sec, 1.5)
        self.assertEqual(records[0].status, "ok")
        self.assertEqual(records[0].ce_value, 5.2)
        self.assertEqual(records[0].pq_value, 6.3)

    def test_unreadable_wav_is_skipped(self) -> None:
        runtime = SimpleNamespace(metric_version="v1")

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            wav_path = model_dir / "speaker01_00001.wav"
            wav_path.write_bytes(b"wav")

            with mock.patch("tts_eval.audiobox.load_audio_sample", side_effect=SkipUtteranceError("unreadable wav: bad header")):
                records, total_audio_sec, n_utts = evaluate_model("model_a", model_dir, runtime, "2026-03-06T00:00:00Z")

        self.assertEqual(n_utts, 1)
        self.assertEqual(total_audio_sec, 0.0)
        self.assertEqual(records[0].status, "skip")
        self.assertEqual(records[0].error, "unreadable wav: bad header")

    def test_backend_failure_marks_record_failed(self) -> None:
        runtime = SimpleNamespace(metric_version="v1")
        waveform = SimpleNamespace(shape=(1, 16000))

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            wav_path = model_dir / "speaker01_00001.wav"
            wav_path.write_bytes(b"wav")

            with mock.patch(
                "tts_eval.audiobox.load_audio_sample",
                return_value=AudioSample(waveform=waveform, sample_rate=16000, duration_sec=1.0),
            ):
                with mock.patch(
                    "tts_eval.audiobox.score_wav_batch_resilient",
                    return_value=({}, {str(wav_path.resolve()): "predict exploded"}),
                ):
                    records, total_audio_sec, n_utts = evaluate_model("model_a", model_dir, runtime, "2026-03-06T00:00:00Z")

        self.assertEqual(n_utts, 1)
        self.assertEqual(total_audio_sec, 1.0)
        self.assertEqual(records[0].status, "fail")
        self.assertEqual(records[0].error, "predict exploded")

    def test_evaluate_model_batches_multiple_utterances(self) -> None:
        runtime = SimpleNamespace(metric_version="v1")
        waveform_a = SimpleNamespace(shape=(1, 16000))
        waveform_b = SimpleNamespace(shape=(1, 18000))

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            wav_a = model_dir / "speaker01_00001.wav"
            wav_b = model_dir / "speaker01_00002.wav"
            wav_a.write_bytes(b"wav")
            wav_b.write_bytes(b"wav")

            load_side_effect = [
                AudioSample(waveform=waveform_a, sample_rate=16000, duration_sec=1.0),
                AudioSample(waveform=waveform_b, sample_rate=16000, duration_sec=1.2),
            ]

            with mock.patch("tts_eval.audiobox.load_audio_sample", side_effect=load_side_effect):
                with mock.patch(
                    "tts_eval.audiobox.score_wav_batch_resilient",
                    return_value=(
                        {
                            str(wav_a.resolve()): (5.1, 6.0),
                            str(wav_b.resolve()): (5.0, 5.9),
                        },
                        {},
                    ),
                ) as score_mock:
                    records, total_audio_sec, n_utts = evaluate_model(
                        "model_a",
                        model_dir,
                        runtime,
                        "2026-03-06T00:00:00Z",
                        batch_size=16,
                    )

        self.assertEqual(n_utts, 2)
        self.assertEqual(total_audio_sec, 2.2)
        self.assertEqual([record.status for record in records], ["ok", "ok"])
        self.assertEqual([record.ce_value for record in records], [5.1, 5.0])
        self.assertEqual([record.pq_value for record in records], [6.0, 5.9])
        self.assertEqual(score_mock.call_count, 1)

    def test_score_wav_batch_resilient_splits_failed_batch(self) -> None:
        runtime = SimpleNamespace(metric_version="v1")
        wav_paths = [Path("/tmp/a.wav"), Path("/tmp/b.wav")]

        def fake_score(batch_paths, runtime, *, batch_size):
            if len(batch_paths) > 1:
                raise RuntimeError("batch exploded")
            if batch_paths[0].name == "a.wav":
                return [(5.2, 6.1)]
            return [(5.0, 5.9)]

        with mock.patch("tts_eval.audiobox.score_wav_batch", side_effect=fake_score):
            scores_by_path, errors_by_path = score_wav_batch_resilient(
                wav_paths,
                runtime,
                batch_size=16,
            )

        self.assertEqual(scores_by_path[str(wav_paths[0].resolve())], (5.2, 6.1))
        self.assertEqual(scores_by_path[str(wav_paths[1].resolve())], (5.0, 5.9))
        self.assertEqual(errors_by_path, {})


class AudioboxRuntimeTests(unittest.TestCase):
    def test_load_runtime_sets_requested_cuda_index(self) -> None:
        fake_audiobox_pkg = ModuleType("audiobox_aesthetics")
        fake_audiobox_pkg.__version__ = "0.0.4"
        fake_infer = ModuleType("audiobox_aesthetics.infer")
        fake_torchaudio = ModuleType("torchaudio")
        fake_set_device = mock.Mock()
        fake_torch = ModuleType("torch")
        fake_torch.cuda = SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 4,
            set_device=fake_set_device,
        )

        class FakePredictor:
            device = "cuda"

        fake_infer.initialize_predictor = lambda: FakePredictor()

        with mock.patch.dict(
            "sys.modules",
            {
                "audiobox_aesthetics": fake_audiobox_pkg,
                "audiobox_aesthetics.infer": fake_infer,
                "torch": fake_torch,
                "torchaudio": fake_torchaudio,
            },
        ):
            with mock.patch("tts_eval.audiobox.package_version", return_value="0.0.4"):
                runtime = load_audiobox_runtime("cuda:1")

        fake_set_device.assert_called_once_with(1)
        self.assertEqual(runtime.execution_device, "cuda:1")
        self.assertIn("audiobox_aesthetics_0.0.4", runtime.metric_version)

    def test_load_runtime_rejects_non_cuda_device(self) -> None:
        with self.assertRaises(ValueError):
            load_audiobox_runtime("cpu")


class AudioboxAudioProbeTests(unittest.TestCase):
    def test_load_audio_sample_uses_torchaudio_info_duration(self) -> None:
        runtime = SimpleNamespace(
            torchaudio=SimpleNamespace(
                info=lambda _: SimpleNamespace(sample_rate=16000, num_frames=32000)
            )
        )

        sample = load_audio_sample(Path("/tmp/test.wav"), runtime)

        self.assertEqual(sample.sample_rate, 16000)
        self.assertEqual(sample.duration_sec, 2.0)
        self.assertIsNone(sample.waveform)

    def test_load_audio_sample_skips_empty_probe(self) -> None:
        runtime = SimpleNamespace(
            torchaudio=SimpleNamespace(
                info=lambda _: SimpleNamespace(sample_rate=16000, num_frames=0)
            )
        )

        with self.assertRaises(SkipUtteranceError):
            load_audio_sample(Path("/tmp/test.wav"), runtime)


if __name__ == "__main__":
    unittest.main()
