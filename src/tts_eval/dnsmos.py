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
    execution_device: str
    num_threads: int | None


def load_dnsmos_runtime(
    personalized: bool = False,
    execution_device: str | None = None,
    num_threads: int | None = None,
) -> DNSMOSRuntime:
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
    resolved_execution_device = execution_device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    return DNSMOSRuntime(
        torch=torch,
        torchaudio=torchaudio,
        functional=deep_noise_suppression_mean_opinion_score,
        metric_version=metric_version,
        personalized=personalized,
        execution_device=resolved_execution_device,
        num_threads=num_threads,
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
    scores = extract_batch_overall_dnsmos(result, expected_count=1)
    return scores[0]


def _collect_score_rows(result: Any) -> list[list[float]]:
    if hasattr(result, "detach"):
        result = result.detach().cpu()
    if hasattr(result, "tolist"):
        result = result.tolist()

    rows: list[list[float]] = []

    def walk(node: Any) -> None:
        if not isinstance(node, list):
            return
        if len(node) >= 4 and all(isinstance(value, (int, float)) for value in node):
            rows.append([float(value) for value in node])
            return
        for child in node:
            walk(child)

    walk(result)
    return rows


def extract_batch_overall_dnsmos(result: Any, expected_count: int) -> list[float]:
    rows = _collect_score_rows(result)
    if len(rows) != expected_count:
        raise RuntimeError("dnsmos result did not expose the expected [batch, ..., 4] shape")

    scores: list[float] = []
    for row in rows:
        overall = row[3]
        if not isinstance(overall, (int, float)):
            raise RuntimeError("dnsmos overall score is not numeric")
        scores.append(float(overall))
    return scores


def score_audio_sample(audio: AudioSample, runtime: DNSMOSRuntime) -> float:
    result = runtime.functional(
        audio.waveform,
        fs=audio.sample_rate,
        personalized=runtime.personalized,
        device=runtime.execution_device,
        num_threads=runtime.num_threads,
    )
    return extract_overall_dnsmos(result)


def score_audio_batch(audios: list[AudioSample], runtime: DNSMOSRuntime) -> list[float]:
    if not audios:
        return []

    sample_rate = audios[0].sample_rate
    if any(audio.sample_rate != sample_rate for audio in audios):
        raise RuntimeError("dnsmos batch must use a single sample rate")
    num_samples = int(audios[0].waveform.shape[-1])
    if any(int(audio.waveform.shape[-1]) != num_samples for audio in audios):
        raise RuntimeError("dnsmos batch must use a single waveform length")

    batch = runtime.torch.stack([audio.waveform.squeeze(0) for audio in audios], dim=0)

    result = runtime.functional(
        batch,
        fs=sample_rate,
        personalized=runtime.personalized,
        device=runtime.execution_device,
        num_threads=runtime.num_threads,
    )
    return extract_batch_overall_dnsmos(result, expected_count=len(audios))


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


def evaluate_model(
    model: str,
    model_dir: Path,
    runtime: DNSMOSRuntime,
    run_timestamp_utc: str,
    batch_size: int = 8,
) -> tuple[list[MetricRecord], float, int]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    records: list[MetricRecord] = []
    total_audio_sec = 0.0
    wav_paths = list(iter_wavs(model_dir))
    n_utts = len(wav_paths)
    pending_utterances: list[Any] = []
    pending_audios: list[AudioSample] = []
    pending_sample_rate: int | None = None
    pending_num_samples: int | None = None

    def flush_pending_batch() -> None:
        nonlocal pending_sample_rate
        nonlocal pending_num_samples
        if not pending_utterances:
            return

        try:
            scores = score_audio_batch(pending_audios, runtime)
            for utterance, score in zip(pending_utterances, scores):
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
        except Exception:
            for utterance, audio in zip(pending_utterances, pending_audios):
                try:
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
        pending_utterances.clear()
        pending_audios.clear()
        pending_sample_rate = None
        pending_num_samples = None

    for wav_path in _iter_wavs_with_progress(model, wav_paths):
        utterance = build_utterance_input(wav_path)

        try:
            audio = load_audio_sample(utterance.wav_path, runtime)
            total_audio_sec += audio.duration_sec
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
        else:
            current_num_samples = int(audio.waveform.shape[-1])
            should_flush = (
                pending_utterances
                and pending_sample_rate is not None
                and pending_num_samples is not None
                and (audio.sample_rate != pending_sample_rate or current_num_samples != pending_num_samples)
            )
            if should_flush:
                flush_pending_batch()

            if pending_sample_rate is None:
                pending_sample_rate = audio.sample_rate
                pending_num_samples = current_num_samples
            pending_utterances.append(utterance)
            pending_audios.append(audio)

            if len(pending_utterances) >= batch_size:
                flush_pending_batch()

    flush_pending_batch()
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
