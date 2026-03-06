from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ModelInput:
    model: str
    model_dir: Path


@dataclass(frozen=True)
class UtteranceInput:
    utt_id: str
    wav_path: Path
    txt_path: Path


def iter_models(inputs_dir: Path) -> list[ModelInput]:
    return [
        ModelInput(model=path.name, model_dir=path)
        for path in sorted(inputs_dir.iterdir())
        if path.is_dir()
    ]


def iter_wavs(model_dir: Path) -> Iterable[Path]:
    yield from sorted(path for path in model_dir.iterdir() if path.is_file() and path.suffix.lower() == ".wav")


def build_utterance_input(wav_path: Path) -> UtteranceInput:
    return UtteranceInput(
        utt_id=wav_path.stem,
        wav_path=wav_path,
        txt_path=wav_path.with_suffix(".txt"),
    )


def parse_speaker_id(utt_id: str) -> str:
    return utt_id.rsplit("_", 1)[0]
