from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tts_eval.chatterbox_turbo import synthesize_request, validate_reference_wavs
from tts_eval.model_runner_inputs import GenerationRequest, SpeakerReferences


class ChatterboxTurboSynthesisTests(unittest.TestCase):
    def test_synthesize_request_caches_conditionals_per_reference(self) -> None:
        saves: list[tuple[str, object, int]] = []

        class NullContextManager:
            def __enter__(self) -> None:
                return None

            def __exit__(self, exc_type, exc, tb) -> bool:  # type: ignore[no-untyped-def]
                del exc_type, exc, tb
                return False

        class FakeChatterbox:
            def __init__(self) -> None:
                self.conds = None
                self.prepare_conditionals_calls: list[str] = []
                self.generate_calls: list[tuple[str, str | None]] = []

            def prepare_conditionals(self, wav_fpath: str, exaggeration: float, norm_loudness: bool) -> None:
                del exaggeration, norm_loudness
                self.prepare_conditionals_calls.append(wav_fpath)
                self.conds = SimpleNamespace(ref=wav_fpath)

            def generate(self, text: str, audio_prompt_path: str | None = None, **kwargs):  # type: ignore[no-untyped-def]
                del kwargs
                self.generate_calls.append((text, audio_prompt_path))
                return SimpleNamespace(cpu=lambda: "waveform")

        chatterbox = FakeChatterbox()
        runtime = SimpleNamespace(
            chatterbox=chatterbox,
            torch=SimpleNamespace(inference_mode=lambda: NullContextManager()),
            torchaudio=SimpleNamespace(save=lambda path, waveform, sample_rate: saves.append((path, waveform, sample_rate))),
            sample_rate=24000,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            request_a = GenerationRequest(
                speaker_id="speaker01",
                utterance_index=1,
                utterance_id="speaker01_00001",
                text="first",
                reference_path=output_dir / "speaker01_ref.wav",
                wav_output_path=output_dir / "speaker01_00001.wav",
                txt_output_path=output_dir / "speaker01_00001.txt",
            )
            request_b = GenerationRequest(
                speaker_id="speaker01",
                utterance_index=2,
                utterance_id="speaker01_00002",
                text="second",
                reference_path=output_dir / "speaker01_ref.wav",
                wav_output_path=output_dir / "speaker01_00002.wav",
                txt_output_path=output_dir / "speaker01_00002.txt",
            )

            conditionals_cache: dict[Path, object] = {}
            synthesize_request(request_a, runtime, conditionals_cache, temperature=0.8, top_p=0.95, top_k=1000, repetition_penalty=1.2)
            synthesize_request(request_b, runtime, conditionals_cache, temperature=0.8, top_p=0.95, top_k=1000, repetition_penalty=1.2)

            first_text = request_a.txt_output_path.read_text(encoding="utf-8")
            second_text = request_b.txt_output_path.read_text(encoding="utf-8")

        self.assertEqual(chatterbox.prepare_conditionals_calls, [str(request_a.reference_path)])
        self.assertEqual(chatterbox.generate_calls, [("first", None), ("second", None)])
        self.assertEqual(first_text, "first\n")
        self.assertEqual(second_text, "second\n")
        self.assertEqual(len(saves), 2)


class ChatterboxTurboReferenceValidationTests(unittest.TestCase):
    def test_validate_reference_wavs_rejects_empty_ref_sets(self) -> None:
        with self.assertRaisesRegex(ValueError, "no reference wavs found"):
            validate_reference_wavs([])

    def test_validate_reference_wavs_rejects_short_reference_audio(self) -> None:
        fake_soundfile = SimpleNamespace(info=lambda path: SimpleNamespace(frames=5 * 24000, samplerate=24000))

        with mock.patch.dict(sys.modules, {"soundfile": fake_soundfile}):
            with self.assertRaisesRegex(ValueError, "longer than 5 seconds"):
                validate_reference_wavs([SpeakerReferences("speaker01", (Path("/refs/speaker01_00001.wav"),))])


if __name__ == "__main__":
    unittest.main()
