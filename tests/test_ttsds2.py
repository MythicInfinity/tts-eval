from __future__ import annotations

import os
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tts_eval.ttsds2 import (
    TTSDS2_WEIGHTS,
    _repair_ttsds_noise_reference_cache,
    _ttsds_package_version,
    collect_valid_wavs,
    evaluate_model,
    inspect_wav,
    run_ttsds2_benchmark,
)


def _write_wav(path: Path, frame_count: int = 160) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * frame_count)


class WavInspectionTests(unittest.TestCase):
    def test_inspect_wav_returns_duration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "sample.wav"
            _write_wav(wav_path, frame_count=320)
            info = inspect_wav(wav_path)
        self.assertAlmostEqual(info.duration_sec, 0.02)

    def test_collect_valid_wavs_skips_invalid_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_dir = Path(tmpdir)
            _write_wav(wav_dir / "valid.wav")
            (wav_dir / "bad.wav").write_bytes(b"not-a-wav")
            valid, skip_count = collect_valid_wavs(wav_dir)
        self.assertEqual(len(valid), 1)
        self.assertEqual(skip_count, 1)


class TTSDS2EvaluationTests(unittest.TestCase):
    def test_ttsds_package_version_uses_module_attribute_when_present(self) -> None:
        self.assertEqual(_ttsds_package_version(SimpleNamespace(__version__="2.1.1")), "2.1.1")

    def test_ttsds_package_version_falls_back_to_package_metadata(self) -> None:
        with mock.patch("tts_eval.ttsds2.metadata.version", return_value="2.1.1"):
            self.assertEqual(_ttsds_package_version(SimpleNamespace()), "2.1.1")

    def test_repair_ttsds_noise_reference_cache_removes_lfs_pointer_tarballs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            noise_reference_dir = cache_dir / "noise-reference"
            noise_reference_dir.mkdir(parents=True)
            (noise_reference_dir / "noise_all_ones.tar.gz").write_text(
                "version https://git-lfs.github.com/spec/v1\n"
                "oid sha256:deadbeef\n"
                "size 123\n",
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, {"TTSDS_CACHE_DIR": str(cache_dir)}):
                _repair_ttsds_noise_reference_cache()

            self.assertFalse(noise_reference_dir.exists())

    def test_repair_ttsds_noise_reference_cache_keeps_real_tarballs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            noise_reference_dir = cache_dir / "noise-reference"
            noise_reference_dir.mkdir(parents=True)
            (noise_reference_dir / "noise_all_ones.tar.gz").write_bytes(b"\x1f\x8b\x08\x00")
            with mock.patch.dict(os.environ, {"TTSDS_CACHE_DIR": str(cache_dir)}):
                _repair_ttsds_noise_reference_cache()

            self.assertTrue(noise_reference_dir.exists())

    def test_run_ttsds2_benchmark_uses_fixed_weights(self) -> None:
        captured: dict[str, object] = {}

        class FakeSuite:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def run(self):
                captured["ran"] = True

            def get_aggregated_results(self):
                return {"generated": {"total": 0.75, "generic": 0.7}}

        runtime = SimpleNamespace(
            DirectoryDataset=lambda path, name=None: {"path": path, "name": name},
            BenchmarkSuite=FakeSuite,
            BenchmarkCategory=SimpleNamespace(
                SPEAKER="speaker",
                INTELLIGIBILITY="intelligibility",
                PROSODY="prosody",
                GENERIC="generic",
                ENVIRONMENT="environment",
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            generated_dir = Path(tmpdir) / "generated"
            refs_dir = Path(tmpdir) / "refs"
            generated_dir.mkdir()
            refs_dir.mkdir()
            result = run_ttsds2_benchmark(generated_dir, refs_dir, runtime)

        self.assertEqual(result.metric_value, 0.75)
        self.assertEqual(result.category_scores, {"generic": 0.7})
        self.assertEqual(captured["datasets"], [{"path": str(generated_dir), "name": "generated"}])
        self.assertEqual(captured["reference_datasets"], [{"path": str(refs_dir), "name": "reference"}])
        self.assertEqual(captured["category_weights"]["speaker"], TTSDS2_WEIGHTS["SPEAKER"])
        self.assertEqual(captured["category_weights"]["generic"], TTSDS2_WEIGHTS["GENERIC"])
        self.assertFalse(captured["include_environment"])
        self.assertTrue(captured["skip_errors"])
        self.assertTrue(captured["ran"])

    def test_evaluate_model_writes_model_level_summary(self) -> None:
        runtime = SimpleNamespace(metric_version="ttsds_v1", package_version="1.0")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "model"
            refs_dir = root / "refs"
            model_dir.mkdir()
            refs_dir.mkdir()
            _write_wav(model_dir / "speaker01_00001.wav", frame_count=1600)
            _write_wav(refs_dir / "speaker01_00001.wav", frame_count=800)

            fake_result = SimpleNamespace(metric_value=0.88, category_scores={"prosody": 0.9}, raw_result={"total": 0.88})
            with mock.patch("tts_eval.ttsds2.run_ttsds2_benchmark", return_value=fake_result):
                summary, metadata = evaluate_model("model_a", model_dir, refs_dir, runtime, "2026-03-06T00:00:00Z")

        self.assertEqual(summary["metric_name"], "ttsds2_total")
        self.assertEqual(summary["metric_value"], 0.88)
        self.assertEqual(summary["fail_count"], 0)
        self.assertEqual(summary["skip_count"], 0)
        self.assertEqual(summary["n_utts"], 1)
        self.assertAlmostEqual(summary["total_audio_sec"], 0.1)
        self.assertEqual(summary["category_scores"], {"prosody": 0.9})
        self.assertEqual(metadata["raw_result"], {"total": 0.88})

    def test_evaluate_model_handles_empty_generated_set(self) -> None:
        runtime = SimpleNamespace(metric_version="ttsds_v1", package_version="1.0")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "model"
            refs_dir = root / "refs"
            model_dir.mkdir()
            refs_dir.mkdir()
            _write_wav(refs_dir / "speaker01_00001.wav")

            summary, metadata = evaluate_model("model_a", model_dir, refs_dir, runtime, "2026-03-06T00:00:00Z")

        self.assertIsNone(summary["metric_value"])
        self.assertEqual(summary["error"], "no valid generated wavs")
        self.assertEqual(summary["fail_count"], 0)
        self.assertEqual(metadata["generated_valid_wavs"], 0)


if __name__ == "__main__":
    unittest.main()
