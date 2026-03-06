from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tts_eval.speaker_sim import (
    AudioSample,
    SkipUtteranceError,
    _patch_hf_hub_download_auth_token_compat,
    evaluate_model,
    index_reference_wavs,
)


class SpeakerReferenceIndexTests(unittest.TestCase):
    def test_index_reference_wavs_groups_paths_by_speaker(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            refs_dir = Path(tmpdir)
            (refs_dir / "speaker01_00001.wav").write_bytes(b"wav")
            (refs_dir / "speaker01_00002.wav").write_bytes(b"wav")
            (refs_dir / "speaker02_00001.wav").write_bytes(b"wav")

            refs_by_speaker = index_reference_wavs(refs_dir)

        self.assertEqual(sorted(refs_by_speaker), ["speaker01", "speaker02"])
        self.assertEqual([path.name for path in refs_by_speaker["speaker01"]], ["speaker01_00001.wav", "speaker01_00002.wav"])


class HuggingFaceHubCompatTests(unittest.TestCase):
    def test_patch_maps_use_auth_token_to_token_for_newer_hub_versions(self) -> None:
        calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def hf_hub_download(*args: object, token: object = None, **kwargs: object) -> str:
            calls.append((args, {"token": token, **kwargs}))
            return "ok"

        module = SimpleNamespace(hf_hub_download=hf_hub_download)

        _patch_hf_hub_download_auth_token_compat(module)

        result = module.hf_hub_download("repo", "file", use_auth_token="secret")

        self.assertEqual(result, "ok")
        self.assertEqual(calls, [(("repo", "file"), {"token": "secret"})])

    def test_patch_converts_missing_entry_errors_to_value_error(self) -> None:
        class EntryNotFoundError(Exception):
            pass

        def hf_hub_download(*args: object, token: object = None, **kwargs: object) -> str:
            raise EntryNotFoundError("custom.py is missing")

        module = SimpleNamespace(hf_hub_download=hf_hub_download)

        _patch_hf_hub_download_auth_token_compat(module)

        with self.assertRaisesRegex(ValueError, "File not found on HF hub"):
            module.hf_hub_download("repo", "custom.py", use_auth_token=False)

    def test_patch_does_not_wrap_when_legacy_kwarg_is_already_supported(self) -> None:
        def hf_hub_download(*args: object, use_auth_token: object = None, **kwargs: object) -> str:
            return "ok"

        module = SimpleNamespace(hf_hub_download=hf_hub_download)

        _patch_hf_hub_download_auth_token_compat(module)

        self.assertIs(module.hf_hub_download, hf_hub_download)
        self.assertIn("use_auth_token", inspect.signature(module.hf_hub_download).parameters)


class SpeakerSimilarityEvaluationTests(unittest.TestCase):
    def test_evaluate_model_scores_matching_speaker_and_reuses_reference_embedding(self) -> None:
        runtime = SimpleNamespace(metric_version="v1")
        waveform = SimpleNamespace()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "model"
            refs_dir = root / "refs"
            model_dir.mkdir()
            refs_dir.mkdir()
            (model_dir / "speaker01_00001.wav").write_bytes(b"wav")
            (model_dir / "speaker01_00002.wav").write_bytes(b"wav")
            (refs_dir / "speaker01_00010.wav").write_bytes(b"wav")

            load_side_effect = [
                AudioSample(waveform=waveform, sample_rate=16000, duration_sec=1.0),
                AudioSample(waveform=waveform, sample_rate=16000, duration_sec=1.5),
            ]
            with mock.patch("tts_eval.speaker_sim.load_audio_sample", side_effect=load_side_effect):
                with mock.patch("tts_eval.speaker_sim.build_reference_embedding", return_value="ref-embedding") as ref_mock:
                    with mock.patch("tts_eval.speaker_sim.score_audio_sample", side_effect=[0.81, 0.79]):
                        records, total_audio_sec, n_utts = evaluate_model(
                            "model_a",
                            model_dir,
                            refs_dir,
                            runtime,
                            "2026-03-06T00:00:00Z",
                            reference_embedding_cache={},
                        )

        self.assertEqual(n_utts, 2)
        self.assertEqual(total_audio_sec, 2.5)
        self.assertEqual([record.metric_value for record in records], [0.81, 0.79])
        self.assertEqual([record.status for record in records], ["ok", "ok"])
        self.assertEqual(ref_mock.call_count, 1)

    def test_evaluate_model_skips_missing_reference_speaker(self) -> None:
        runtime = SimpleNamespace(metric_version="v1")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "model"
            refs_dir = root / "refs"
            model_dir.mkdir()
            refs_dir.mkdir()
            (model_dir / "speaker02_00001.wav").write_bytes(b"wav")
            (refs_dir / "speaker01_00001.wav").write_bytes(b"wav")

            with mock.patch(
                "tts_eval.speaker_sim.load_audio_sample",
                return_value=AudioSample(waveform=SimpleNamespace(), sample_rate=16000, duration_sec=1.25),
            ):
                records, total_audio_sec, n_utts = evaluate_model("model_a", model_dir, refs_dir, runtime, "2026-03-06T00:00:00Z")

        self.assertEqual(n_utts, 1)
        self.assertEqual(total_audio_sec, 1.25)
        self.assertEqual(records[0].status, "skip")
        self.assertEqual(records[0].error, "missing reference speaker wavs for speaker02")

    def test_unreadable_generated_wav_is_skipped(self) -> None:
        runtime = SimpleNamespace(metric_version="v1")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "model"
            refs_dir = root / "refs"
            model_dir.mkdir()
            refs_dir.mkdir()
            (model_dir / "speaker01_00001.wav").write_bytes(b"wav")
            (refs_dir / "speaker01_00002.wav").write_bytes(b"wav")

            with mock.patch("tts_eval.speaker_sim.load_audio_sample", side_effect=SkipUtteranceError("unreadable wav: bad header")):
                records, total_audio_sec, n_utts = evaluate_model("model_a", model_dir, refs_dir, runtime, "2026-03-06T00:00:00Z")

        self.assertEqual(n_utts, 1)
        self.assertEqual(total_audio_sec, 0.0)
        self.assertEqual(records[0].status, "skip")
        self.assertEqual(records[0].error, "unreadable wav: bad header")

    def test_failed_reference_embedding_is_memoized_per_speaker(self) -> None:
        runtime = SimpleNamespace(metric_version="v1")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "model"
            refs_dir = root / "refs"
            model_dir.mkdir()
            refs_dir.mkdir()
            (model_dir / "speaker01_00001.wav").write_bytes(b"wav")
            (model_dir / "speaker01_00002.wav").write_bytes(b"wav")
            (refs_dir / "speaker01_00010.wav").write_bytes(b"wav")

            load_side_effect = [
                AudioSample(waveform=SimpleNamespace(), sample_rate=16000, duration_sec=1.0),
                AudioSample(waveform=SimpleNamespace(), sample_rate=16000, duration_sec=1.5),
            ]
            with mock.patch("tts_eval.speaker_sim.load_audio_sample", side_effect=load_side_effect):
                with mock.patch(
                    "tts_eval.speaker_sim.build_reference_embedding",
                    side_effect=SkipUtteranceError("no valid reference wavs for speaker speaker01"),
                ) as ref_mock:
                    records, total_audio_sec, n_utts = evaluate_model(
                        "model_a",
                        model_dir,
                        refs_dir,
                        runtime,
                        "2026-03-06T00:00:00Z",
                        reference_embedding_cache={},
                    )

        self.assertEqual(n_utts, 2)
        self.assertEqual(total_audio_sec, 2.5)
        self.assertEqual([record.status for record in records], ["skip", "skip"])
        self.assertEqual(
            [record.error for record in records],
            ["no valid reference wavs for speaker speaker01", "no valid reference wavs for speaker speaker01"],
        )
        self.assertEqual(ref_mock.call_count, 1)
