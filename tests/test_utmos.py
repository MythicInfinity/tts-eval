from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tts_eval.utmos import (
    AudioSample,
    SkipUtteranceError,
    batch_prediction_seed,
    build_prediction_batch_aliases,
    evaluate_model,
    extract_batch_predictions,
    extract_scalar_prediction,
    score_wav_batch_resilient,
)


class UTMOSHelpersTests(unittest.TestCase):
    def test_extract_scalar_prediction_accepts_float(self) -> None:
        self.assertEqual(extract_scalar_prediction(4.2), 4.2)

    def test_extract_scalar_prediction_accepts_singleton_list(self) -> None:
        self.assertEqual(extract_scalar_prediction([4.2]), 4.2)

    def test_extract_scalar_prediction_rejects_non_scalar(self) -> None:
        with self.assertRaises(RuntimeError):
            extract_scalar_prediction([4.2, 4.1])

    def test_extract_batch_predictions_maps_scores_back_to_input_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_a = Path(tmpdir) / "speaker01_00001.wav"
            wav_b = Path(tmpdir) / "speaker01_00002.wav"
            wav_a.write_bytes(b"wav")
            wav_b.write_bytes(b"wav")
            alias_a = Path(tmpdir) / "alias_a.wav"
            alias_b = Path(tmpdir) / "alias_b.wav"
            alias_a.write_bytes(b"wav")
            alias_b.write_bytes(b"wav")

            scores = extract_batch_predictions(
                [
                    {"file_path": str(alias_b), "predicted_mos": 4.0},
                    {"file_path": str(alias_a), "predicted_mos": 4.2},
                ],
                {
                    str(alias_a.resolve()): wav_a,
                    str(alias_b.resolve()): wav_b,
                },
                [wav_a, wav_b],
            )

        self.assertEqual(scores, [4.2, 4.0])

    def test_batch_prediction_seed_is_deterministic(self) -> None:
        runtime = SimpleNamespace(config="fusion_stage3", fold=0, seed=42)
        wav_paths = [Path("/tmp/a.wav"), Path("/tmp/b.wav")]
        self.assertEqual(batch_prediction_seed(runtime, wav_paths), batch_prediction_seed(runtime, list(reversed(wav_paths))))

    def test_build_prediction_batch_aliases_normalizes_uppercase_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "speaker01_00001.WAV"
            wav_path.write_bytes(b"wav")

            batch_dir, alias_to_original, val_list = build_prediction_batch_aliases([wav_path])
            try:
                alias_paths = list(Path(batch_dir.name).glob("*.wav"))
                self.assertEqual(len(alias_paths), 1)
                self.assertEqual(val_list, ["utt_00000"])
                self.assertEqual(alias_to_original[str(alias_paths[0].resolve())], wav_path)
            finally:
                batch_dir.cleanup()


class UTMOSEvaluationTests(unittest.TestCase):
    def test_evaluate_model_accumulates_meaningful_audio_duration(self) -> None:
        runtime = SimpleNamespace(metric_version="v1")
        waveform = SimpleNamespace(shape=(1, 16000))

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            wav_path = model_dir / "speaker01_00001.wav"
            wav_path.write_bytes(b"wav")

            with mock.patch("tts_eval.utmos.load_audio_sample", return_value=AudioSample(waveform=waveform, sample_rate=16000, duration_sec=1.5)):
                with mock.patch("tts_eval.utmos.score_wav_batch", return_value=[4.05]):
                    records, total_audio_sec, n_utts = evaluate_model("model_a", model_dir, runtime, "2026-03-06T00:00:00Z")

        self.assertEqual(n_utts, 1)
        self.assertEqual(total_audio_sec, 1.5)
        self.assertEqual(records[0].status, "ok")
        self.assertEqual(records[0].metric_value, 4.05)

    def test_unreadable_wav_is_skipped(self) -> None:
        runtime = SimpleNamespace(metric_version="v1")

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            wav_path = model_dir / "speaker01_00001.wav"
            wav_path.write_bytes(b"wav")

            with mock.patch("tts_eval.utmos.load_audio_sample", side_effect=SkipUtteranceError("unreadable wav: bad header")):
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

            with mock.patch("tts_eval.utmos.load_audio_sample", return_value=AudioSample(waveform=waveform, sample_rate=16000, duration_sec=1.0)):
                with mock.patch("tts_eval.utmos.score_wav_batch", side_effect=RuntimeError("predict exploded")):
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
            (model_dir / "speaker01_00001.wav").write_bytes(b"wav")
            (model_dir / "speaker01_00002.wav").write_bytes(b"wav")

            load_side_effect = [
                AudioSample(waveform=waveform_a, sample_rate=16000, duration_sec=1.0),
                AudioSample(waveform=waveform_b, sample_rate=16000, duration_sec=1.2),
            ]

            with mock.patch("tts_eval.utmos.load_audio_sample", side_effect=load_side_effect):
                with mock.patch("tts_eval.utmos.score_wav_batch", return_value=[4.0, 4.1]) as batch_mock:
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
        self.assertEqual([record.metric_value for record in records], [4.0, 4.1])
        self.assertEqual(batch_mock.call_count, 1)

    def test_score_wav_batch_resilient_splits_failed_batch(self) -> None:
        runtime = SimpleNamespace(metric_version="v1")
        wav_paths = [Path("/tmp/a.wav"), Path("/tmp/b.wav")]

        def fake_score(batch_paths, model_dir, runtime, *, batch_size, num_workers):
            if len(batch_paths) > 1:
                raise RuntimeError("batch exploded")
            return [4.0 if batch_paths[0].name == "a.wav" else 4.1]

        with mock.patch("tts_eval.utmos.score_wav_batch", side_effect=fake_score):
            scores_by_path, errors_by_path = score_wav_batch_resilient(
                wav_paths,
                Path("/tmp"),
                runtime,
                batch_size=16,
                num_workers=0,
            )

        self.assertEqual(scores_by_path[str(wav_paths[0].resolve())], 4.0)
        self.assertEqual(scores_by_path[str(wav_paths[1].resolve())], 4.1)
        self.assertEqual(errors_by_path, {})


if __name__ == "__main__":
    unittest.main()
