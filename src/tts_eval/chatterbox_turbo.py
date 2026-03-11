from __future__ import annotations

import copy
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from tts_eval.io import ensure_dir
from tts_eval.model_runner_inputs import GenerationRequest, SpeakerReferences

CHATTERBOX_TURBO_MODEL_ID = "ResembleAI/chatterbox-turbo"


@dataclass(frozen=True)
class ChatterboxTurboRuntime:
    chatterbox: Any
    torch: Any
    torchaudio: Any
    device: str
    sample_rate: int
    model_id: str
    package_version: str
    metric_version: str


def load_chatterbox_turbo_runtime(device: str) -> ChatterboxTurboRuntime:
    try:
        import torch
        import torchaudio
        from chatterbox.tts_turbo import ChatterboxTurboTTS
    except ModuleNotFoundError as exc:
        raise RuntimeError("chatterbox-tts, torch, and torchaudio must be installed in the runner environment") from exc

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device was requested but is not available")

    chatterbox = ChatterboxTurboTTS.from_pretrained(device=device)
    package_version = metadata.version("chatterbox-tts")
    metric_version = f"chatterbox_turbo|package:{package_version}|model:{CHATTERBOX_TURBO_MODEL_ID}"
    return ChatterboxTurboRuntime(
        chatterbox=chatterbox,
        torch=torch,
        torchaudio=torchaudio,
        device=device,
        sample_rate=chatterbox.sr,
        model_id=CHATTERBOX_TURBO_MODEL_ID,
        package_version=package_version,
        metric_version=metric_version,
    )


def validate_reference_wavs(speaker_references: list[SpeakerReferences]) -> list[SpeakerReferences]:
    if not speaker_references:
        raise ValueError("no reference wavs found")

    try:
        import soundfile
    except ModuleNotFoundError as exc:
        raise RuntimeError("soundfile must be installed in the runner environment") from exc

    for speaker_reference in speaker_references:
        for wav_path in speaker_reference.wav_paths:
            try:
                info = soundfile.info(str(wav_path))
            except Exception as exc:
                raise ValueError(f"unreadable reference wav {wav_path}: {exc}") from exc

            if info.frames <= 0 or info.samplerate <= 0:
                raise ValueError(f"empty reference wav {wav_path}")

            duration_sec = info.frames / info.samplerate
            if duration_sec <= 5.0:
                raise ValueError(
                    f"reference wav must be longer than 5 seconds for Chatterbox Turbo: {wav_path}"
                )

    return speaker_references


def _assign_cached_conditionals(
    request: GenerationRequest,
    runtime: ChatterboxTurboRuntime,
    conditionals_cache: dict[Path, Any],
) -> None:
    cached_conditionals = conditionals_cache.get(request.reference_path)
    if cached_conditionals is None:
        runtime.chatterbox.prepare_conditionals(str(request.reference_path), exaggeration=0.0, norm_loudness=True)
        cached_conditionals = copy.deepcopy(runtime.chatterbox.conds)
        conditionals_cache[request.reference_path] = cached_conditionals

    runtime.chatterbox.conds = copy.deepcopy(cached_conditionals)


def synthesize_request(
    request: GenerationRequest,
    runtime: ChatterboxTurboRuntime,
    conditionals_cache: dict[Path, Any],
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> None:
    _assign_cached_conditionals(request, runtime, conditionals_cache)
    with runtime.torch.inference_mode():
        waveform = runtime.chatterbox.generate(
            request.text,
            audio_prompt_path=None,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            cfg_weight=0.0,
            exaggeration=0.0,
            min_p=0.0,
            norm_loudness=True,
        )

    ensure_dir(request.wav_output_path.parent)
    runtime.torchaudio.save(str(request.wav_output_path), waveform.cpu(), runtime.sample_rate)
    request.txt_output_path.write_text(request.text + "\n", encoding="utf-8")
