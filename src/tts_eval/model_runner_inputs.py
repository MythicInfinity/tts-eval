from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from tts_eval.discovery import iter_wavs, parse_speaker_id

TARGET_UTTERANCES_PER_SPEAKER = 128


class UtteranceTextExhaustedError(RuntimeError):
    """Raised when the global utterance text stream runs out before planning completes."""


@dataclass(frozen=True)
class SpeakerReferences:
    speaker_id: str
    wav_paths: tuple[Path, ...]


@dataclass(frozen=True)
class GenerationRequest:
    speaker_id: str
    utterance_index: int
    utterance_id: str
    text: str
    reference_path: Path
    wav_output_path: Path
    txt_output_path: Path


def index_speaker_references(refs_dir: Path) -> list[SpeakerReferences]:
    refs_by_speaker: dict[str, list[Path]] = {}
    for wav_path in iter_wavs(refs_dir):
        speaker_id = parse_speaker_id(wav_path.stem)
        refs_by_speaker.setdefault(speaker_id, []).append(wav_path)

    return [
        SpeakerReferences(speaker_id=speaker_id, wav_paths=tuple(sorted(wav_paths)))
        for speaker_id, wav_paths in sorted(refs_by_speaker.items())
    ]


def _next_nonempty_text(text_iter: Iterator[str]) -> str:
    for text in text_iter:
        normalized = text.strip()
        if normalized:
            return normalized
    raise UtteranceTextExhaustedError("utterance text stream ended before all speakers reached the target count")


def count_generation_requests(
    speaker_references: Iterable[SpeakerReferences],
    target_utterances_per_speaker: int = TARGET_UTTERANCES_PER_SPEAKER,
) -> int:
    if target_utterances_per_speaker <= 0:
        raise ValueError("target_utterances_per_speaker must be positive")

    materialized = list(speaker_references)
    if not materialized:
        return 0
    return len(materialized) * target_utterances_per_speaker


def build_generation_requests(
    speaker_references: Iterable[SpeakerReferences],
    utterance_texts: Iterable[str],
    output_dir: Path,
    target_utterances_per_speaker: int = TARGET_UTTERANCES_PER_SPEAKER,
) -> Iterator[GenerationRequest]:
    if target_utterances_per_speaker <= 0:
        raise ValueError("target_utterances_per_speaker must be positive")

    speaker_references = list(speaker_references)
    if not speaker_references:
        raise ValueError("no reference wavs found")

    text_iter = iter(utterance_texts)

    for speaker_reference in speaker_references:
        for utterance_index in range(1, target_utterances_per_speaker + 1):
            text = _next_nonempty_text(text_iter)
            reference_path = speaker_reference.wav_paths[(utterance_index - 1) % len(speaker_reference.wav_paths)]
            utterance_id = f"{speaker_reference.speaker_id}_{utterance_index:05d}"
            wav_output_path = output_dir / f"{utterance_id}.wav"
            txt_output_path = output_dir / f"{utterance_id}.txt"
            yield GenerationRequest(
                speaker_id=speaker_reference.speaker_id,
                utterance_index=utterance_index,
                utterance_id=utterance_id,
                text=text,
                reference_path=reference_path,
                wav_output_path=wav_output_path,
                txt_output_path=txt_output_path,
            )
