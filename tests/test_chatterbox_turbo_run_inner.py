from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


def _load_run_inner_module():
    module_path = Path("/home/iss/code/tts-eval-chatterbox-turbo/model-runners/chatterbox_turbo/run_inner.py")
    spec = importlib.util.spec_from_file_location("chatterbox_turbo_run_inner_test", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load chatterbox turbo run_inner module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ChatterboxTurboRunInnerTests(unittest.TestCase):
    def test_main_validates_dataset_before_loading_runtime(self) -> None:
        module = _load_run_inner_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            refs_dir = Path(tmpdir) / "refs"
            output_dir = Path(tmpdir) / "out"
            refs_dir.mkdir()
            argv = [
                "run_inner.py",
                "--refs",
                str(refs_dir),
                "--output",
                str(output_dir),
            ]

            with mock.patch.object(module.sys, "argv", argv):
                with mock.patch.object(
                    module,
                    "build_utterance_text_dataset",
                    side_effect=ValueError("no utterance text datasets configured"),
                ):
                    with mock.patch.object(module, "load_chatterbox_turbo_runtime") as runtime_mock:
                        with self.assertRaisesRegex(ValueError, "no utterance text datasets configured"):
                            module.main()

        runtime_mock.assert_not_called()

    def test_main_validates_references_before_loading_runtime(self) -> None:
        module = _load_run_inner_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            refs_dir = Path(tmpdir) / "refs"
            output_dir = Path(tmpdir) / "out"
            refs_dir.mkdir()
            argv = [
                "run_inner.py",
                "--refs",
                str(refs_dir),
                "--output",
                str(output_dir),
            ]

            with mock.patch.object(module.sys, "argv", argv):
                with mock.patch.object(module, "build_utterance_text_dataset", return_value=iter(["hello"])):
                    with mock.patch.object(module, "index_speaker_references", return_value=[]):
                        with mock.patch.object(
                            module,
                            "validate_reference_wavs",
                            side_effect=ValueError("no reference wavs found"),
                        ):
                            with mock.patch.object(module, "load_chatterbox_turbo_runtime") as runtime_mock:
                                with self.assertRaisesRegex(ValueError, "no reference wavs found"):
                                    module.main()

        runtime_mock.assert_not_called()

    def test_main_cleans_up_staging_dir_when_request_iteration_fails(self) -> None:
        module = _load_run_inner_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            refs_dir = root / "refs"
            refs_dir.mkdir()
            output_dir = root / "chatterbox_turbo"
            argv = [
                "run_inner.py",
                "--refs",
                str(refs_dir),
                "--output",
                str(output_dir),
            ]
            staged_paths: list[Path] = []

            def fake_build_staging_dir(final_output_dir: Path) -> Path:
                staged_dir = final_output_dir.parent / ".staged"
                staged_dir.mkdir(parents=True, exist_ok=True)
                staged_paths.append(staged_dir)
                return staged_dir

            def failing_requests(*args, **kwargs):  # type: ignore[no-untyped-def]
                del args, kwargs
                raise RuntimeError("boom")
                yield  # pragma: no cover

            with mock.patch.object(module.sys, "argv", argv):
                with mock.patch.object(module, "build_utterance_text_dataset", return_value=iter(["hello"])):
                    with mock.patch.object(
                        module,
                        "index_speaker_references",
                        return_value=[SimpleNamespace(speaker_id="speaker01", wav_paths=(refs_dir / "speaker01_00001.wav",))],
                    ):
                        with mock.patch.object(
                            module,
                            "validate_reference_wavs",
                            return_value=[SimpleNamespace(speaker_id="speaker01", wav_paths=(refs_dir / "speaker01_00001.wav",))],
                        ):
                            with mock.patch.object(module, "count_generation_requests", return_value=1):
                                with mock.patch.object(module, "load_chatterbox_turbo_runtime", return_value=SimpleNamespace()):
                                    with mock.patch.object(module, "_build_staging_dir", side_effect=fake_build_staging_dir):
                                        with mock.patch.object(module, "build_generation_requests", side_effect=failing_requests):
                                            with self.assertRaisesRegex(RuntimeError, "boom"):
                                                module.main()

            self.assertEqual(len(staged_paths), 1)
            self.assertFalse(staged_paths[0].exists())


if __name__ == "__main__":
    unittest.main()
