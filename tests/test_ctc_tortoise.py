from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tts_eval.ctc import AudioSample, SkipUtteranceError
from tts_eval.ctc_tortoise import (
    _SYMBOL_TO_ID,
    decode_greedy,
    english_cleaners,
    evaluate_model,
    text_to_sequence,
    tokenize_transcript,
    validate_vocab_alignment,
)


class TortoiseTokenizerTests(unittest.TestCase):
    def test_english_cleaners_expand_common_abbreviations(self) -> None:
        cleaned = english_cleaners("Dr. Smith met Mr. Jones on St. Patrick street.")
        self.assertEqual(cleaned, "doctor smith met mister jones on saint patrick street.")

    def test_text_to_sequence_keeps_punctuation_symbols(self) -> None:
        sequence = text_to_sequence("TTS, actually.")
        self.assertEqual(tokenize_transcript("TTS, actually.").normalized_text, "tts, actually.")
        self.assertGreater(len(sequence), 0)

    def test_text_to_sequence_supports_tortoise_arpabet_tokens_when_vocab_provided(self) -> None:
        vocab = {"|": 11, "t": 57, "u": 58, "r": 55, "n": 51, "@HH": 106, "@AW1": 82, ".": 7}
        sequence = text_to_sequence("turn {HH AW1}.", symbol_to_id=vocab)
        self.assertEqual(sequence, [57, 58, 55, 51, 11, 106, 82, 7])

    def test_validate_vocab_alignment_accepts_hf_pad_alias(self) -> None:
        vocab = {
            ("<pad>" if symbol == "_" else "|" if symbol == " " else symbol): token_id
            for symbol, token_id in _SYMBOL_TO_ID.items()
        }
        validate_vocab_alignment(vocab)

    def test_tokenize_transcript_filters_tokenizer_ids_outside_model_vocab(self) -> None:
        class FakeTokenizer:
            def __call__(self, text: str, add_special_tokens: bool = False) -> SimpleNamespace:
                self.last_text = text
                self.last_add_special_tokens = add_special_tokens
                return SimpleNamespace(input_ids=[57, 11, 150, 42])

        tokenizer = FakeTokenizer()
        tokenized = tokenize_transcript("T e", tokenizer=tokenizer, model_vocab_size=148)
        self.assertEqual(tokenized.normalized_text, "t e")
        self.assertEqual(tokenized.token_ids, (57, 11, 42))

    def test_tokenize_transcript_skips_when_empty_after_cleaning(self) -> None:
        with self.assertRaises(SkipUtteranceError):
            tokenize_transcript("[]{}~")

    def test_decode_greedy_collapses_repeats_and_blanks(self) -> None:
        labels = ("_", "-", " ", "a", "b")
        decoded = decode_greedy([0, 3, 3, 0, 2, 4, 4], labels, blank_id=0)
        self.assertEqual(decoded, "a b")

    def test_decode_greedy_skips_special_tokens_and_out_of_range_ids(self) -> None:
        labels = ("<pad>", "-", " ", "a", "b", "<unk>")
        decoded = decode_greedy([0, 3, 5, 99, 4], labels, blank_id=0, skip_token_ids={5})
        self.assertEqual(decoded, "ab")


class TortoiseEvaluateModelTests(unittest.TestCase):
    def test_missing_transcript_still_counts_audio_duration(self) -> None:
        runtime = SimpleNamespace(metric_version="v1")

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            wav_path = model_dir / "speaker01_00001.wav"
            wav_path.write_bytes(b"not-a-real-wav")

            with mock.patch("tts_eval.ctc_tortoise.load_audio_sample", return_value=AudioSample(waveform=None, sample_rate=16000, duration_sec=1.25)):
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

            with mock.patch("tts_eval.ctc_tortoise.load_audio_sample", side_effect=SkipUtteranceError("unreadable wav: bad header")):
                records, total_audio_sec, n_utts = evaluate_model("model_a", model_dir, runtime, "2026-03-06T00:00:00Z")

        self.assertEqual(n_utts, 1)
        self.assertEqual(total_audio_sec, 0.0)
        self.assertEqual(records[0].status, "skip")
        self.assertEqual(records[0].error, "unreadable wav: bad header")


if __name__ == "__main__":
    unittest.main()
