from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tts_eval.model_runner_inputs import (
    TARGET_UTTERANCES_PER_SPEAKER,
    UtteranceTextExhaustedError,
    build_generation_requests,
    count_generation_requests,
    index_speaker_references,
)


class SpeakerReferenceIndexTests(unittest.TestCase):
    def test_index_speaker_references_sorts_speakers_and_wavs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            refs_dir = Path(tmpdir)
            (refs_dir / "speaker02_00002.wav").write_bytes(b"wav")
            (refs_dir / "speaker01_00002.wav").write_bytes(b"wav")
            (refs_dir / "speaker01_00001.wav").write_bytes(b"wav")

            indexed = index_speaker_references(refs_dir)

        self.assertEqual([item.speaker_id for item in indexed], ["speaker01", "speaker02"])
        self.assertEqual([path.name for path in indexed[0].wav_paths], ["speaker01_00001.wav", "speaker01_00002.wav"])


class GenerationRequestPlanningTests(unittest.TestCase):
    def test_generation_requests_are_lazy_over_text_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            refs_dir = Path(tmpdir)
            output_dir = refs_dir / "out"
            (refs_dir / "speaker01_00001.wav").write_bytes(b"wav")

            consumed: list[str] = []

            def text_stream():
                for value in ("first", "second"):
                    consumed.append(value)
                    yield value

            requests = build_generation_requests(
                index_speaker_references(refs_dir),
                text_stream(),
                output_dir,
                target_utterances_per_speaker=2,
            )

            first_request = next(requests)

        self.assertEqual(first_request.text, "first")
        self.assertEqual(consumed, ["first"])

    def test_round_robins_references_per_speaker(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            refs_dir = root / "refs"
            output_dir = root / "out"
            refs_dir.mkdir()
            (refs_dir / "speaker01_00001.wav").write_bytes(b"wav")
            (refs_dir / "speaker01_00002.wav").write_bytes(b"wav")

            requests = list(build_generation_requests(
                index_speaker_references(refs_dir),
                ["one", "two", "three", "four"],
                output_dir,
                target_utterances_per_speaker=4,
            ))

        self.assertEqual([request.reference_path.name for request in requests], [
            "speaker01_00001.wav",
            "speaker01_00002.wav",
            "speaker01_00001.wav",
            "speaker01_00002.wav",
        ])

    def test_global_text_stream_does_not_reset_between_speakers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            refs_dir = root / "refs"
            output_dir = root / "out"
            refs_dir.mkdir()
            (refs_dir / "speaker01_00001.wav").write_bytes(b"wav")
            (refs_dir / "speaker02_00001.wav").write_bytes(b"wav")

            requests = list(build_generation_requests(
                index_speaker_references(refs_dir),
                ["text-1", "text-2", "text-3", "text-4"],
                output_dir,
                target_utterances_per_speaker=2,
            ))

        self.assertEqual([request.text for request in requests], ["text-1", "text-2", "text-3", "text-4"])
        self.assertEqual([request.utterance_id for request in requests], [
            "speaker01_00001",
            "speaker01_00002",
            "speaker02_00001",
            "speaker02_00002",
        ])

    def test_raises_when_text_stream_is_exhausted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            refs_dir = Path(tmpdir)
            (refs_dir / "speaker01_00001.wav").write_bytes(b"wav")
            (refs_dir / "speaker02_00001.wav").write_bytes(b"wav")

            with self.assertRaises(UtteranceTextExhaustedError):
                list(
                    build_generation_requests(
                        index_speaker_references(refs_dir),
                        ["only-one"],
                        refs_dir / "out",
                        target_utterances_per_speaker=1,
                    )
                )

    def test_default_target_constant_is_expected_value(self) -> None:
        self.assertEqual(TARGET_UTTERANCES_PER_SPEAKER, 128)

    def test_count_generation_requests_uses_speaker_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            refs_dir = Path(tmpdir)
            (refs_dir / "speaker01_00001.wav").write_bytes(b"wav")
            (refs_dir / "speaker02_00001.wav").write_bytes(b"wav")

            total = count_generation_requests(index_speaker_references(refs_dir), target_utterances_per_speaker=3)

        self.assertEqual(total, 6)


if __name__ == "__main__":
    unittest.main()
