from __future__ import annotations

import json
import math
import platform
import re
import statistics
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, Iterable

from tts_eval.discovery import build_utterance_input, iter_wavs
from tts_eval.io import ensure_dir


class SkipUtteranceError(Exception):
    """Raised when an utterance should be skipped rather than failed."""


@dataclass(frozen=True)
class AudioSample:
    waveform: Any | None
    sample_rate: int
    duration_sec: float


@dataclass(frozen=True)
class AudioboxRuntime:
    torch: Any
    torchaudio: Any
    predictor: Any
    metric_version: str
    execution_device: str


@dataclass(frozen=True)
class AudioboxRecord:
    run_timestamp_utc: str
    metric_name: str
    metric_version: str
    model: str
    utt_id: str
    wav_path: str
    ce_value: float | None
    pq_value: float | None
    status: str
    error: str | None


def _normalize_path_key(path: str | Path) -> str:
    return str(Path(path).resolve())


def load_audiobox_runtime(execution_device: str | None = None) -> AudioboxRuntime:
    requested_device = execution_device or "cuda"
    if not re.fullmatch(r"cuda(?::\d+)?", requested_device):
        raise ValueError("Audiobox runner currently supports only --device cuda or cuda:<index>")

    requested_index: int | None = None
    if ":" in requested_device:
        requested_index = int(requested_device.split(":", 1)[1])

    try:
        import audiobox_aesthetics
        import torch
        import torchaudio
        from audiobox_aesthetics.infer import initialize_predictor
    except ModuleNotFoundError as exc:
        raise RuntimeError("audiobox_aesthetics, torch, and torchaudio must be installed in the runner environment") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required but not available")

    if requested_index is not None:
        if requested_index < 0 or requested_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"requested CUDA device index {requested_index} is out of range "
                f"(available={torch.cuda.device_count()})"
            )
        # Predictor loads model via .to(\"cuda\"); setting current device pins that to the requested index.
        torch.cuda.set_device(requested_index)

    predictor = initialize_predictor()
    predictor_device = getattr(predictor, "device", None)
    if predictor_device is not None and "cuda" not in str(predictor_device):
        raise RuntimeError(f"audiobox predictor was not initialized on CUDA: {predictor_device}")

    if predictor_device is None:
        resolved_execution_device = requested_device
    else:
        predictor_device_str = str(predictor_device)
        if predictor_device_str == "cuda" and requested_index is not None:
            resolved_execution_device = requested_device
        else:
            resolved_execution_device = predictor_device_str

    resolved_version = None
    for dist_name in ("audiobox-aesthetics", "audiobox_aesthetics"):
        try:
            resolved_version = package_version(dist_name)
            break
        except PackageNotFoundError:
            continue
    if resolved_version is None:
        resolved_version = getattr(audiobox_aesthetics, "__version__", "unknown")

    metric_version = (
        f"audiobox_aesthetics_{resolved_version}"
        f"|axes=CE,PQ"
        f"|predict=forward_path_batch"
    )

    return AudioboxRuntime(
        torch=torch,
        torchaudio=torchaudio,
        predictor=predictor,
        metric_version=metric_version,
        execution_device=resolved_execution_device,
    )


def load_audio_sample(path: Path, runtime: AudioboxRuntime) -> AudioSample:
    try:
        info = runtime.torchaudio.info(str(path))
    except Exception as exc:
        raise SkipUtteranceError(f"unreadable wav: {exc}") from exc

    sample_rate = int(getattr(info, "sample_rate", 0) or 0)
    num_frames = int(getattr(info, "num_frames", 0) or 0)
    duration_sec = num_frames / sample_rate if sample_rate else 0.0
    if duration_sec <= 0:
        raise SkipUtteranceError("empty wav")

    return AudioSample(waveform=None, sample_rate=sample_rate, duration_sec=duration_sec)


def _extract_axis_value(row: dict[str, Any], axis: str) -> float:
    value = row.get(axis)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"audiobox output field {axis} is not numeric")
    numeric = float(value)
    if math.isnan(numeric):
        raise RuntimeError(f"audiobox output field {axis} resolved to NaN")
    return numeric


def extract_batch_scores(result: Any, expected_count: int) -> list[tuple[float, float]]:
    if not isinstance(result, list) or len(result) != expected_count:
        raise RuntimeError("audiobox prediction did not return one result per wav")

    scores: list[tuple[float, float]] = []
    for row in result:
        if not isinstance(row, dict):
            raise RuntimeError("audiobox prediction row is not a dict")
        ce = _extract_axis_value(row, "CE")
        pq = _extract_axis_value(row, "PQ")
        scores.append((ce, pq))
    return scores


def score_wav_batch(wav_paths: list[Path], runtime: AudioboxRuntime, *, batch_size: int) -> list[tuple[float, float]]:
    if not wav_paths:
        return []

    # The upstream predictor performs one forward pass over the provided batch list.
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    request_batch = [{"path": str(wav_path)} for wav_path in wav_paths]
    prediction = runtime.predictor.forward(request_batch)
    return extract_batch_scores(prediction, expected_count=len(wav_paths))


def score_wav_batch_resilient(
    wav_paths: list[Path],
    runtime: AudioboxRuntime,
    *,
    batch_size: int,
) -> tuple[dict[str, tuple[float, float]], dict[str, str]]:
    if not wav_paths:
        return {}, {}

    try:
        scores = score_wav_batch(wav_paths, runtime, batch_size=min(batch_size, len(wav_paths)))
        return ({_normalize_path_key(wav_path): score for wav_path, score in zip(wav_paths, scores, strict=True)}, {})
    except Exception as exc:
        if len(wav_paths) == 1:
            return {}, {_normalize_path_key(wav_paths[0]): str(exc)}

        midpoint = len(wav_paths) // 2
        left_scores, left_errors = score_wav_batch_resilient(
            wav_paths[:midpoint],
            runtime,
            batch_size=batch_size,
        )
        right_scores, right_errors = score_wav_batch_resilient(
            wav_paths[midpoint:],
            runtime,
            batch_size=batch_size,
        )
        return left_scores | right_scores, left_errors | right_errors


def _build_progress_bar(model: str, total: int) -> Any | None:
    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError:
        return None

    return tqdm(
        total=total,
        desc=f"[audiobox] utts model={model}",
        unit="utt",
        dynamic_ncols=True,
        mininterval=1.0,
    )


def evaluate_model(
    model: str,
    model_dir: Path,
    runtime: AudioboxRuntime,
    run_timestamp_utc: str,
    *,
    batch_size: int = 16,
) -> tuple[list[AudioboxRecord], float, int]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    records: list[AudioboxRecord] = []
    total_audio_sec = 0.0
    wav_paths = list(iter_wavs(model_dir))
    n_utts = len(wav_paths)
    pending_utterances: list[Any] = []
    pending_wav_paths: list[Path] = []
    progress_bar = _build_progress_bar(model, n_utts)
    ok_count = 0
    skip_count = 0
    fail_count = 0

    def append_record(record: AudioboxRecord) -> None:
        nonlocal ok_count, skip_count, fail_count
        records.append(record)
        if record.status == "ok":
            ok_count += 1
        elif record.status == "skip":
            skip_count += 1
        elif record.status == "fail":
            fail_count += 1
        if progress_bar is not None:
            progress_bar.update(1)
            progress_bar.set_postfix(ok=ok_count, skip=skip_count, fail=fail_count, refresh=False)

    def flush_pending_batch() -> None:
        if not pending_utterances:
            return

        scores_by_path, errors_by_path = score_wav_batch_resilient(
            pending_wav_paths,
            runtime,
            batch_size=batch_size,
        )

        for utterance in pending_utterances:
            normalized_path = _normalize_path_key(utterance.wav_path)
            if normalized_path in scores_by_path:
                ce_value, pq_value = scores_by_path[normalized_path]
                append_record(
                    AudioboxRecord(
                        run_timestamp_utc=run_timestamp_utc,
                        metric_name="audiobox_ce_pq",
                        metric_version=runtime.metric_version,
                        model=model,
                        utt_id=utterance.utt_id,
                        wav_path=str(utterance.wav_path),
                        ce_value=ce_value,
                        pq_value=pq_value,
                        status="ok",
                        error=None,
                    )
                )
            elif normalized_path in errors_by_path:
                append_record(
                    AudioboxRecord(
                        run_timestamp_utc=run_timestamp_utc,
                        metric_name="audiobox_ce_pq",
                        metric_version=runtime.metric_version,
                        model=model,
                        utt_id=utterance.utt_id,
                        wav_path=str(utterance.wav_path),
                        ce_value=None,
                        pq_value=None,
                        status="fail",
                        error=errors_by_path[normalized_path],
                    )
                )
            else:  # pragma: no cover - defensive guard if resilient scoring returns an incomplete mapping
                append_record(
                    AudioboxRecord(
                        run_timestamp_utc=run_timestamp_utc,
                        metric_name="audiobox_ce_pq",
                        metric_version=runtime.metric_version,
                        model=model,
                        utt_id=utterance.utt_id,
                        wav_path=str(utterance.wav_path),
                        ce_value=None,
                        pq_value=None,
                        status="fail",
                        error="audiobox batch scoring did not return a result for this utterance",
                    )
                )

        pending_utterances.clear()
        pending_wav_paths.clear()

    try:
        for wav_path in wav_paths:
            utterance = build_utterance_input(wav_path)

            try:
                audio = load_audio_sample(utterance.wav_path, runtime)
                total_audio_sec += audio.duration_sec
            except SkipUtteranceError as exc:
                append_record(
                    AudioboxRecord(
                        run_timestamp_utc=run_timestamp_utc,
                        metric_name="audiobox_ce_pq",
                        metric_version=runtime.metric_version,
                        model=model,
                        utt_id=utterance.utt_id,
                        wav_path=str(utterance.wav_path),
                        ce_value=None,
                        pq_value=None,
                        status="skip",
                        error=str(exc),
                    )
                )
                continue
            except Exception as exc:  # pragma: no cover - exercised with backend/runtime failures
                append_record(
                    AudioboxRecord(
                        run_timestamp_utc=run_timestamp_utc,
                        metric_name="audiobox_ce_pq",
                        metric_version=runtime.metric_version,
                        model=model,
                        utt_id=utterance.utt_id,
                        wav_path=str(utterance.wav_path),
                        ce_value=None,
                        pq_value=None,
                        status="fail",
                        error=str(exc),
                    )
                )
                continue

            pending_utterances.append(utterance)
            pending_wav_paths.append(utterance.wav_path)
            if len(pending_utterances) >= batch_size:
                flush_pending_batch()

        flush_pending_batch()
    finally:
        if progress_bar is not None:
            progress_bar.close()

    return records, total_audio_sec, n_utts


def _summarize_axis(values: list[float]) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    mean = statistics.fmean(values)
    median = statistics.median(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0

    if math.isnan(mean) or math.isnan(median) or math.isnan(std):
        raise ValueError("audiobox summary statistics contain NaN")

    return mean, median, std


def build_summary_payload(
    run_timestamp_utc: str,
    metric_version: str,
    model: str,
    n_utts: int,
    total_audio_sec: float,
    records: list[AudioboxRecord],
) -> dict[str, Any]:
    ce_values = [record.ce_value for record in records if record.status == "ok" and record.ce_value is not None]
    pq_values = [record.pq_value for record in records if record.status == "ok" and record.pq_value is not None]

    ce_mean, ce_median, ce_std = _summarize_axis(ce_values)
    pq_mean, pq_median, pq_std = _summarize_axis(pq_values)

    fail_count = sum(record.status == "fail" for record in records)
    skip_count = sum(record.status == "skip" for record in records)

    return {
        "run_timestamp_utc": run_timestamp_utc,
        "metric_name": "audiobox_ce_pq",
        "metric_version": metric_version,
        "model": model,
        "n_utts": n_utts,
        "total_audio_sec": round(total_audio_sec, 6),
        "ce_mean": ce_mean,
        "ce_median": ce_median,
        "ce_std": ce_std,
        "pq_mean": pq_mean,
        "pq_median": pq_median,
        "pq_std": pq_std,
        "fail_count": fail_count,
        "skip_count": skip_count,
    }


def build_metadata_payload(metric_version: str) -> dict[str, Any]:
    return {
        "metric_name": "audiobox_ce_pq",
        "metric_version": metric_version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "axes": ["CE", "PQ"],
        "prediction_mode": "predictor.forward(path_batch)",
        "device_policy": "cuda_only",
    }


def write_audiobox_jsonl(path: Path, records: Iterable[AudioboxRecord]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=False) + "\n")
