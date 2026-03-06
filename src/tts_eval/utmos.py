from __future__ import annotations

import hashlib
import math
import platform
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tts_eval.discovery import build_utterance_input, iter_wavs
from tts_eval.io import MetricRecord
from tts_eval.stats import aggregate_metric_records


UTMOS_DEFAULT_CONFIG = "fusion_stage3"
UTMOS_DEFAULT_FOLD = 0
UTMOS_DEFAULT_SEED = 42


class SkipUtteranceError(Exception):
    """Raised when an utterance should be skipped rather than failed."""


@dataclass(frozen=True)
class AudioSample:
    waveform: Any | None
    sample_rate: int
    duration_sec: float


@dataclass(frozen=True)
class UTMOSRuntime:
    torch: Any
    torchaudio: Any
    utmosv2: Any
    model: Any
    metric_version: str
    execution_device: str
    config: str
    fold: int
    seed: int
    remove_silent_section: bool
    predict_dataset: str
    num_repetitions: int


def load_utmos_runtime(
    execution_device: str | None = None,
    *,
    config: str = UTMOS_DEFAULT_CONFIG,
    fold: int = UTMOS_DEFAULT_FOLD,
    seed: int = UTMOS_DEFAULT_SEED,
    remove_silent_section: bool = True,
    predict_dataset: str = "sarulab",
    num_repetitions: int = 1,
) -> UTMOSRuntime:
    try:
        import torch
        import torchaudio
        import utmosv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch, torchaudio, and utmosv2 must be installed in the runner environment") from exc

    resolved_execution_device = execution_device or "cuda:0"
    if not resolved_execution_device.startswith("cuda"):
        raise ValueError("UTMOS runner currently supports only CUDA devices")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required but not available")
    model = utmosv2.create_model(
        pretrained=True,
        config=config,
        fold=fold,
        seed=seed,
        device=resolved_execution_device,
    )
    metric_version = (
        f"utmosv2_{utmosv2.__version__}"
        f"|config:{config}"
        f"|fold={fold}"
        f"|seed={seed}"
        f"|predict=input_dir"
        f"|predict_dataset={predict_dataset}"
        f"|num_repetitions={num_repetitions}"
        f"|remove_silent_section={str(remove_silent_section).lower()}"
    )
    return UTMOSRuntime(
        torch=torch,
        torchaudio=torchaudio,
        utmosv2=utmosv2,
        model=model,
        metric_version=metric_version,
        execution_device=resolved_execution_device,
        config=config,
        fold=fold,
        seed=seed,
        remove_silent_section=remove_silent_section,
        predict_dataset=predict_dataset,
        num_repetitions=num_repetitions,
    )


def load_audio_sample(path: Path, runtime: UTMOSRuntime) -> AudioSample:
    # UTMOS scoring is path-based, so we only need lightweight metadata here.
    # This avoids fully decoding every file in Python before UTMOS decodes it again.
    try:
        info = runtime.torchaudio.info(str(path))
    except Exception as exc:
        raise SkipUtteranceError(f"unreadable wav: {exc}") from exc

    sample_rate = int(getattr(info, "sample_rate", 0) or 0)
    num_frames = int(getattr(info, "num_frames", 0) or 0)
    num_channels = int(getattr(info, "num_channels", 0) or 0)

    if sample_rate <= 0:
        raise SkipUtteranceError("unreadable wav: invalid sample rate")
    if num_frames <= 0:
        raise SkipUtteranceError("empty wav")
    if num_channels <= 0:
        raise SkipUtteranceError("unexpected waveform rank")

    duration_sec = num_frames / sample_rate
    if duration_sec <= 0:
        raise SkipUtteranceError("empty wav")

    return AudioSample(waveform=None, sample_rate=sample_rate, duration_sec=duration_sec)


def extract_scalar_prediction(prediction: Any) -> float:
    if isinstance(prediction, bool):
        raise RuntimeError("utmos prediction is boolean, expected scalar MOS")
    if isinstance(prediction, (int, float)):
        value = float(prediction)
    else:
        if hasattr(prediction, "detach"):
            prediction = prediction.detach().cpu()
        if hasattr(prediction, "numel") and prediction.numel() == 1 and hasattr(prediction, "item"):
            value = float(prediction.item())
        else:
            if hasattr(prediction, "tolist"):
                prediction = prediction.tolist()
            while isinstance(prediction, (list, tuple)) and len(prediction) == 1:
                prediction = prediction[0]
            if not isinstance(prediction, (int, float)) or isinstance(prediction, bool):
                raise RuntimeError("utmos prediction did not resolve to a single numeric value")
            value = float(prediction)

    if math.isnan(value):
        raise RuntimeError("utmos prediction resolved to NaN")
    return value


def _normalize_path_key(path: str | Path) -> str:
    return str(Path(path).resolve())


def batch_prediction_seed(runtime: UTMOSRuntime, wav_paths: list[Path]) -> int:
    material = "|".join(
        [
            runtime.config,
            str(runtime.fold),
            str(runtime.seed),
            *sorted(_normalize_path_key(wav_path) for wav_path in wav_paths),
        ]
    )
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def seed_prediction_batch(runtime: UTMOSRuntime, wav_paths: list[Path]) -> int:
    seed = batch_prediction_seed(runtime, wav_paths)
    random.seed(seed)
    runtime.torch.manual_seed(seed)
    if hasattr(runtime.torch, "cuda") and runtime.torch.cuda.is_available():
        runtime.torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
    except ModuleNotFoundError:
        pass
    else:
        np.random.seed(seed)
    return seed


def build_prediction_batch_aliases(wav_paths: list[Path]) -> tuple[tempfile.TemporaryDirectory[str], dict[str, Path], list[str]]:
    batch_dir = tempfile.TemporaryDirectory(prefix="utmos-batch-")
    alias_to_original: dict[str, Path] = {}
    val_list: list[str] = []

    for index, wav_path in enumerate(wav_paths):
        alias_stem = f"utt_{index:05d}"
        alias_path = Path(batch_dir.name) / f"{alias_stem}.wav"
        alias_path.symlink_to(wav_path.resolve())
        alias_to_original[_normalize_path_key(alias_path)] = wav_path
        val_list.append(alias_stem)

    return batch_dir, alias_to_original, val_list


def _build_progress_bar(model: str, total: int) -> Any | None:
    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError:
        return None

    return tqdm(
        total=total,
        desc=f"[utmos] utts model={model}",
        unit="utt",
        dynamic_ncols=True,
        mininterval=1.0,
    )


def extract_batch_predictions(predictions: Any, alias_to_original: dict[str, Path], wav_paths: list[Path]) -> list[float]:
    if not isinstance(predictions, list) or len(predictions) != len(wav_paths):
        raise RuntimeError("utmos batch prediction did not return one result per wav")

    scores_by_path: dict[str, float] = {}
    for prediction in predictions:
        if not isinstance(prediction, dict):
            raise RuntimeError("utmos batch prediction item is not a dict")
        file_path = prediction.get("file_path")
        if not isinstance(file_path, str):
            raise RuntimeError("utmos batch prediction item did not include a file_path string")
        score = extract_scalar_prediction(prediction.get("predicted_mos"))
        path_key = _normalize_path_key(file_path)
        original_wav_path = alias_to_original.get(path_key)
        if original_wav_path is None:
            raise RuntimeError(f"utmos batch prediction referenced unknown file_path: {file_path}")
        original_path_key = _normalize_path_key(original_wav_path)
        if original_path_key in scores_by_path:
            raise RuntimeError(f"utmos batch prediction duplicated file_path: {file_path}")
        scores_by_path[original_path_key] = score

    scores: list[float] = []
    for wav_path in wav_paths:
        path_key = _normalize_path_key(wav_path)
        if path_key not in scores_by_path:
            raise RuntimeError(f"utmos batch prediction missing score for: {wav_path}")
        scores.append(scores_by_path[path_key])

    return scores


def score_wav_batch(wav_paths: list[Path], model_dir: Path, runtime: UTMOSRuntime, *, batch_size: int, num_workers: int) -> list[float]:
    if not wav_paths:
        return []

    seed_prediction_batch(runtime, wav_paths)
    batch_dir, alias_to_original, val_list = build_prediction_batch_aliases(wav_paths)
    try:
        prediction = runtime.model.predict(
            input_dir=batch_dir.name,
            val_list=val_list,
            device=runtime.execution_device,
            num_workers=num_workers,
            batch_size=batch_size,
            num_repetitions=runtime.num_repetitions,
            predict_dataset=runtime.predict_dataset,
            remove_silent_section=runtime.remove_silent_section,
            verbose=False,
        )
        return extract_batch_predictions(prediction, alias_to_original, wav_paths)
    finally:
        batch_dir.cleanup()


def score_wav_batch_resilient(
    wav_paths: list[Path],
    model_dir: Path,
    runtime: UTMOSRuntime,
    *,
    batch_size: int,
    num_workers: int,
) -> tuple[dict[str, float], dict[str, str]]:
    if not wav_paths:
        return {}, {}

    try:
        scores = score_wav_batch(
            wav_paths,
            model_dir,
            runtime,
            batch_size=min(batch_size, len(wav_paths)),
            num_workers=num_workers,
        )
        return (
            {_normalize_path_key(wav_path): score for wav_path, score in zip(wav_paths, scores, strict=True)},
            {},
        )
    except Exception as exc:
        if len(wav_paths) == 1:
            return {}, {_normalize_path_key(wav_paths[0]): str(exc)}

        midpoint = len(wav_paths) // 2
        left_scores, left_errors = score_wav_batch_resilient(
            wav_paths[:midpoint],
            model_dir,
            runtime,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        right_scores, right_errors = score_wav_batch_resilient(
            wav_paths[midpoint:],
            model_dir,
            runtime,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        return left_scores | right_scores, left_errors | right_errors


def build_metadata_payload(
    metric_version: str,
    *,
    config: str,
    fold: int,
    seed: int,
    remove_silent_section: bool,
    predict_dataset: str,
    num_repetitions: int,
) -> dict[str, Any]:
    return {
        "metric_name": "utmos",
        "metric_version": metric_version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "model_config": config,
        "fold": fold,
        "seed": seed,
        "pretrained": True,
        "prediction_mode": "input_dir_batch",
        "remove_silent_section": remove_silent_section,
        "predict_dataset": predict_dataset,
        "num_repetitions": num_repetitions,
    }


def evaluate_model(
    model: str,
    model_dir: Path,
    runtime: UTMOSRuntime,
    run_timestamp_utc: str,
    *,
    batch_size: int = 16,
    num_workers: int = 0,
) -> tuple[list[MetricRecord], float, int]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")

    records: list[MetricRecord] = []
    total_audio_sec = 0.0
    wav_paths = list(iter_wavs(model_dir))
    n_utts = len(wav_paths)
    valid_utterances: list[Any] = []
    progress_bar = _build_progress_bar(model, n_utts)
    ok_count = 0
    skip_count = 0
    fail_count = 0

    def append_record(record: MetricRecord) -> None:
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

    try:
        for wav_path in wav_paths:
            utterance = build_utterance_input(wav_path)

            try:
                audio = load_audio_sample(utterance.wav_path, runtime)
                total_audio_sec += audio.duration_sec
            except SkipUtteranceError as exc:
                append_record(
                    MetricRecord(
                        run_timestamp_utc=run_timestamp_utc,
                        metric_name="utmos",
                        metric_version=runtime.metric_version,
                        model=model,
                        utt_id=utterance.utt_id,
                        wav_path=str(utterance.wav_path),
                        metric_value=None,
                        status="skip",
                        error=str(exc),
                    )
                )
                continue
            except Exception as exc:  # pragma: no cover - exercised with backend/runtime failures
                append_record(
                    MetricRecord(
                        run_timestamp_utc=run_timestamp_utc,
                        metric_name="utmos",
                        metric_version=runtime.metric_version,
                        model=model,
                        utt_id=utterance.utt_id,
                        wav_path=str(utterance.wav_path),
                        metric_value=None,
                        status="fail",
                        error=str(exc),
                    )
                )
                continue

            valid_utterances.append(utterance)

        if valid_utterances:
            valid_wav_paths = [utterance.wav_path for utterance in valid_utterances]
            scores_by_path, errors_by_path = score_wav_batch_resilient(
                valid_wav_paths,
                model_dir,
                runtime,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            for utterance in valid_utterances:
                normalized_path = _normalize_path_key(utterance.wav_path)
                if normalized_path in scores_by_path:
                    append_record(
                        MetricRecord(
                            run_timestamp_utc=run_timestamp_utc,
                            metric_name="utmos",
                            metric_version=runtime.metric_version,
                            model=model,
                            utt_id=utterance.utt_id,
                            wav_path=str(utterance.wav_path),
                            metric_value=scores_by_path[normalized_path],
                            status="ok",
                            error=None,
                        )
                    )
                elif normalized_path in errors_by_path:
                    append_record(
                        MetricRecord(
                            run_timestamp_utc=run_timestamp_utc,
                            metric_name="utmos",
                            metric_version=runtime.metric_version,
                            model=model,
                            utt_id=utterance.utt_id,
                            wav_path=str(utterance.wav_path),
                            metric_value=None,
                            status="fail",
                            error=errors_by_path[normalized_path],
                        )
                    )
                else:  # pragma: no cover - defensive guard if resilient scoring returns an incomplete mapping
                    append_record(
                        MetricRecord(
                            run_timestamp_utc=run_timestamp_utc,
                            metric_name="utmos",
                            metric_version=runtime.metric_version,
                            model=model,
                            utt_id=utterance.utt_id,
                            wav_path=str(utterance.wav_path),
                            metric_value=None,
                            status="fail",
                            error="utmos batch scoring did not return a result for this utterance",
                        )
                    )
    finally:
        if progress_bar is not None:
            progress_bar.close()

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
        "metric_name": "utmos",
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
