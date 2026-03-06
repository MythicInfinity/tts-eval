from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tts_eval.discovery import build_utterance_input, iter_wavs
from tts_eval.io import MetricRecord
from tts_eval.stats import aggregate_metric_records


class SkipUtteranceError(Exception):
    """Raised when an utterance should be skipped rather than failed."""


@dataclass(frozen=True)
class AudioSample:
    waveform: Any
    sample_rate: int
    duration_sec: float


@dataclass(frozen=True)
class DNSMOSRuntime:
    torch: Any
    torchaudio: Any
    functional: Any
    metric_version: str
    personalized: bool


def load_dnsmos_runtime(personalized: bool = False) -> DNSMOSRuntime:
    try:
        import torch
        import torchaudio
        from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score
        import torchmetrics
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch, torchaudio, and torchmetrics must be installed in the runner environment") from exc

    metric_version = (
        f"torchmetrics_dnsmos_{torchmetrics.__version__}"
        f"|personalized={str(personalized).lower()}"
        f"|output=ovrl"
    )
    return DNSMOSRuntime(
        torch=torch,
        torchaudio=torchaudio,
        functional=deep_noise_suppression_mean_opinion_score,
        metric_version=metric_version,
        personalized=personalized,
    )


def load_audio_sample(path: Path, runtime: DNSMOSRuntime) -> AudioSample:
    try:
        waveform, sample_rate = runtime.torchaudio.load(str(path))
    except Exception as exc:
        raise SkipUtteranceError(f"unreadable wav: {exc}") from exc

    if waveform.numel() == 0:
        raise SkipUtteranceError("empty wav")

    if waveform.ndim != 2:
        raise SkipUtteranceError("unexpected waveform rank")

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    duration_sec = waveform.shape[-1] / sample_rate if sample_rate else 0.0
    if duration_sec <= 0:
        raise SkipUtteranceError("empty wav")

    return AudioSample(waveform=waveform, sample_rate=sample_rate, duration_sec=duration_sec)


def extract_overall_dnsmos(result: Any) -> float:
    if hasattr(result, "detach"):
        result = result.detach().cpu()
    if hasattr(result, "tolist"):
        result = result.tolist()

    while isinstance(result, list) and len(result) == 1:
        result = result[0]

    if not isinstance(result, list) or len(result) < 4:
        raise RuntimeError("dnsmos result did not expose the expected [..., 4] shape")

    overall = result[3]
    if not isinstance(overall, (int, float)):
        raise RuntimeError("dnsmos overall score is not numeric")
    return float(overall)


def score_audio_sample(audio: AudioSample, runtime: DNSMOSRuntime) -> float:
    result = runtime.functional(
        audio.waveform,
        fs=audio.sample_rate,
        personalized=runtime.personalized,
    )
    return extract_overall_dnsmos(result)


def build_metadata_payload(metric_version: str, personalized: bool) -> dict[str, Any]:
    return {
        "metric_name": "dnsmos_ovrl",
        "metric_version": metric_version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "personalized": personalized,
        "output_component": "ovrl",
    }


def _iter_wavs_with_progress(model: str, wav_paths: list[Path]) -> Any:
    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError:
        return wav_paths

    return tqdm(
        wav_paths,
        desc=f"[dnsmos] utts model={model}",
        total=len(wav_paths),
        unit="utt",
        dynamic_ncols=True,
        mininterval=1.0,
    )


def evaluate_model(model: str, model_dir: Path, runtime: DNSMOSRuntime, run_timestamp_utc: str) -> tuple[list[MetricRecord], float, int]:
    records: list[MetricRecord] = []
    total_audio_sec = 0.0
    wav_paths = list(iter_wavs(model_dir))
    n_utts = len(wav_paths)

    for wav_path in _iter_wavs_with_progress(model, wav_paths):
        utterance = build_utterance_input(wav_path)

        try:
            audio = load_audio_sample(utterance.wav_path, runtime)
            total_audio_sec += audio.duration_sec
            score = score_audio_sample(audio, runtime)
            records.append(
                MetricRecord(
                    run_timestamp_utc=run_timestamp_utc,
                    metric_name="dnsmos_ovrl",
                    metric_version=runtime.metric_version,
                    model=model,
                    utt_id=utterance.utt_id,
                    wav_path=str(utterance.wav_path),
                    metric_value=score,
                    status="ok",
                    error=None,
                )
            )
        except SkipUtteranceError as exc:
            records.append(
                MetricRecord(
                    run_timestamp_utc=run_timestamp_utc,
                    metric_name="dnsmos_ovrl",
                    metric_version=runtime.metric_version,
                    model=model,
                    utt_id=utterance.utt_id,
                    wav_path=str(utterance.wav_path),
                    metric_value=None,
                    status="skip",
                    error=str(exc),
                )
            )
        except Exception as exc:  # pragma: no cover - exercised with backend/runtime failures
            records.append(
                MetricRecord(
                    run_timestamp_utc=run_timestamp_utc,
                    metric_name="dnsmos_ovrl",
                    metric_version=runtime.metric_version,
                    model=model,
                    utt_id=utterance.utt_id,
                    wav_path=str(utterance.wav_path),
                    metric_value=None,
                    status="fail",
                    error=str(exc),
                )
            )

    return records, total_audio_sec, n_utts


def build_summary_payload(
    run_timestamp_utc: str,
    metric_version: str,
    model: str,
    n_utts: int,
    total_audio_sec: float,
    records: list[MetricRecord],
) -> dict[str, Any]:
    aggregate = aggregate_metric_records(records)
    return {
        "run_timestamp_utc": run_timestamp_utc,
        "metric_name": "dnsmos_ovrl",
        "metric_version": metric_version,
        "model": model,
        "n_utts": n_utts,
        "total_audio_sec": round(total_audio_sec, 6),
        "metric_mean": aggregate.metric_mean,
        "metric_median": aggregate.metric_median,
        "metric_std": aggregate.metric_std,
        "fail_count": aggregate.fail_count,
        "skip_count": aggregate.skip_count,
    }
