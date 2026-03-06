from __future__ import annotations

import os
import platform
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tts_eval.discovery import build_utterance_input, iter_wavs, parse_speaker_id
from tts_eval.io import MetricRecord
from tts_eval.stats import aggregate_metric_records


SPEAKER_SIM_CHECKPOINT = "speechbrain/spkrec-ecapa-voxceleb"
REFERENCE_EMBEDDING_ERROR_PREFIX = "__error__:"


class SkipUtteranceError(Exception):
    """Raised when an utterance should be skipped rather than failed."""


@dataclass(frozen=True)
class AudioSample:
    waveform: Any
    sample_rate: int
    duration_sec: float


@dataclass(frozen=True)
class SpeakerSimRuntime:
    torch: Any
    torchaudio: Any
    classifier: Any
    metric_version: str
    execution_device: str
    sample_rate: int
    checkpoint: str


def _patch_hf_hub_download_auth_token_compat(huggingface_hub: Any) -> None:
    hf_hub_download = getattr(huggingface_hub, "hf_hub_download", None)
    if hf_hub_download is None:
        return

    try:
        parameters = inspect.signature(hf_hub_download).parameters
    except (TypeError, ValueError):
        return

    if "use_auth_token" in parameters:
        return

    missing_entry_error_names = {
        "EntryNotFoundError",
        "LocalEntryNotFoundError",
        "RemoteEntryNotFoundError",
    }

    def hf_hub_download_compat(*args: Any, **kwargs: Any) -> Any:
        use_auth_token = kwargs.pop("use_auth_token", None)
        if use_auth_token is not None and "token" not in kwargs:
            kwargs["token"] = use_auth_token
        try:
            return hf_hub_download(*args, **kwargs)
        except Exception as exc:
            if exc.__class__.__name__ in missing_entry_error_names:
                raise ValueError("File not found on HF hub") from exc
            raise

    huggingface_hub.hf_hub_download = hf_hub_download_compat


def load_speaker_sim_runtime(execution_device: str | None = None) -> SpeakerSimRuntime:
    try:
        import huggingface_hub
        import speechbrain
        import torch
        import torchaudio
        from speechbrain.inference.speaker import EncoderClassifier
    except ModuleNotFoundError as exc:
        raise RuntimeError("speechbrain, torch, and torchaudio must be installed in the runner environment") from exc

    _patch_hf_hub_download_auth_token_compat(huggingface_hub)
    resolved_execution_device = execution_device or ("cuda" if torch.cuda.is_available() else "cpu")
    cache_root = Path(os.environ.get("SPEECHBRAIN_CACHE_DIR", str(Path.home() / ".cache" / "speechbrain")))
    savedir = cache_root / "spkrec-ecapa-voxceleb"
    classifier = EncoderClassifier.from_hparams(
        source=SPEAKER_SIM_CHECKPOINT,
        savedir=str(savedir),
        run_opts={"device": resolved_execution_device},
    )
    if hasattr(classifier, "eval"):
        classifier.eval()

    sample_rate = 16000
    hparams = getattr(classifier, "hparams", None)
    classifier_sample_rate = getattr(hparams, "sample_rate", None)
    if isinstance(classifier_sample_rate, (int, float)) and int(classifier_sample_rate) > 0:
        sample_rate = int(classifier_sample_rate)

    metric_version = (
        f"speechbrain_{speechbrain.__version__}"
        f"|checkpoint:{SPEAKER_SIM_CHECKPOINT}"
        f"|sample_rate={sample_rate}"
        f"|ref_pool=mean_embedding"
        f"|score=cosine_similarity"
    )
    return SpeakerSimRuntime(
        torch=torch,
        torchaudio=torchaudio,
        classifier=classifier,
        metric_version=metric_version,
        execution_device=resolved_execution_device,
        sample_rate=sample_rate,
        checkpoint=SPEAKER_SIM_CHECKPOINT,
    )


def load_audio_sample(path: Path, runtime: SpeakerSimRuntime) -> AudioSample:
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

    if sample_rate != runtime.sample_rate:
        waveform = runtime.torchaudio.functional.resample(waveform, sample_rate, runtime.sample_rate)
        sample_rate = runtime.sample_rate

    return AudioSample(waveform=waveform, sample_rate=sample_rate, duration_sec=duration_sec)


def index_reference_wavs(refs_dir: Path) -> dict[str, list[Path]]:
    refs_by_speaker: dict[str, list[Path]] = {}
    for wav_path in iter_wavs(refs_dir):
        speaker_id = parse_speaker_id(wav_path.stem)
        refs_by_speaker.setdefault(speaker_id, []).append(wav_path)
    return refs_by_speaker


def _iter_wavs_with_progress(model: str, wav_paths: list[Path]) -> Any:
    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError:
        return wav_paths

    return tqdm(
        wav_paths,
        desc=f"[speaker_sim] utts model={model}",
        total=len(wav_paths),
        unit="utt",
        dynamic_ncols=True,
        mininterval=1.0,
    )


def extract_embedding_batch(audios: list[AudioSample], runtime: SpeakerSimRuntime) -> Any:
    if not audios:
        return []

    batch = runtime.torch.stack([audio.waveform.squeeze(0) for audio in audios], dim=0)
    with runtime.torch.inference_mode():
        embeddings = runtime.classifier.encode_batch(batch.to(runtime.execution_device))

    if hasattr(embeddings, "detach"):
        embeddings = embeddings.detach()
    if hasattr(embeddings, "reshape"):
        embeddings = embeddings.reshape(len(audios), -1)
    return embeddings


def extract_embedding(audio: AudioSample, runtime: SpeakerSimRuntime) -> Any:
    batch_embedding = extract_embedding_batch([audio], runtime)
    embedding = batch_embedding[0]
    if hasattr(embedding, "reshape"):
        embedding = embedding.reshape(-1)
    return embedding


def build_reference_embedding(
    speaker_id: str,
    reference_paths: list[Path],
    runtime: SpeakerSimRuntime,
    batch_size: int = 8,
) -> Any:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    batched_embeddings: list[Any] = []
    pending_audios: list[AudioSample] = []
    pending_num_samples: int | None = None

    def flush_pending_batch() -> None:
        nonlocal pending_num_samples
        if not pending_audios:
            return
        batched_embeddings.append(extract_embedding_batch(pending_audios, runtime))
        pending_audios.clear()
        pending_num_samples = None

    for reference_path in reference_paths:
        try:
            audio = load_audio_sample(reference_path, runtime)
        except SkipUtteranceError:
            continue
        current_num_samples = int(audio.waveform.shape[-1])
        should_flush = pending_audios and pending_num_samples is not None and current_num_samples != pending_num_samples
        if should_flush:
            flush_pending_batch()

        if pending_num_samples is None:
            pending_num_samples = current_num_samples
        pending_audios.append(audio)
        if len(pending_audios) >= batch_size:
            flush_pending_batch()

    flush_pending_batch()

    if not batched_embeddings:
        raise SkipUtteranceError(f"no valid reference wavs for speaker {speaker_id}")

    return runtime.torch.cat(batched_embeddings, dim=0).mean(dim=0)


def score_audio_batch(audios: list[AudioSample], reference_embeddings: list[Any], runtime: SpeakerSimRuntime) -> list[float]:
    if not audios:
        return []
    if len(audios) != len(reference_embeddings):
        raise RuntimeError("speaker_sim batch requires one reference embedding per utterance")

    generated_embeddings = extract_embedding_batch(audios, runtime)
    reference_batch = runtime.torch.stack(reference_embeddings, dim=0)
    if hasattr(reference_batch, "to"):
        reference_batch = reference_batch.to(runtime.execution_device)

    similarities = runtime.torch.nn.functional.cosine_similarity(
        generated_embeddings.reshape(len(audios), -1),
        reference_batch.reshape(len(reference_embeddings), -1),
        dim=-1,
    )
    if hasattr(similarities, "detach"):
        similarities = similarities.detach().cpu()
    if hasattr(similarities, "tolist"):
        return [float(value) for value in similarities.tolist()]
    return [float(value) for value in similarities]


def score_audio_sample(audio: AudioSample, reference_embedding: Any, runtime: SpeakerSimRuntime) -> float:
    similarities = score_audio_batch([audio], [reference_embedding], runtime)
    return similarities[0]


def build_metadata_payload(metric_version: str) -> dict[str, Any]:
    return {
        "metric_name": "speaker_sim_ecapa",
        "metric_version": metric_version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "checkpoint": SPEAKER_SIM_CHECKPOINT,
        "reference_pooling": "mean_embedding",
        "similarity": "cosine_similarity",
        "batching_policy": "same_waveform_length_only",
    }


def evaluate_model(
    model: str,
    model_dir: Path,
    refs_dir: Path,
    runtime: SpeakerSimRuntime,
    run_timestamp_utc: str,
    reference_wavs_by_speaker: dict[str, list[Path]] | None = None,
    reference_embedding_cache: dict[str, Any] | None = None,
    batch_size: int = 8,
) -> tuple[list[MetricRecord], float, int]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    records: list[MetricRecord] = []
    total_audio_sec = 0.0
    wav_paths = list(iter_wavs(model_dir))
    n_utts = len(wav_paths)
    refs_by_speaker = reference_wavs_by_speaker or index_reference_wavs(refs_dir)
    embedding_cache = reference_embedding_cache if reference_embedding_cache is not None else {}
    pending_utterances: list[Any] = []
    pending_audios: list[AudioSample] = []
    pending_references: list[Any] = []
    pending_num_samples: int | None = None

    def flush_pending_batch() -> None:
        nonlocal pending_num_samples
        if not pending_utterances:
            return
        try:
            similarities = score_audio_batch(pending_audios, pending_references, runtime)
            for utterance, similarity in zip(pending_utterances, similarities, strict=True):
                records.append(
                    MetricRecord(
                        run_timestamp_utc=run_timestamp_utc,
                        metric_name="speaker_sim_ecapa",
                        metric_version=runtime.metric_version,
                        model=model,
                        utt_id=utterance.utt_id,
                        wav_path=str(utterance.wav_path),
                        metric_value=similarity,
                        status="ok",
                        error=None,
                    )
                )
        except Exception as exc:  # pragma: no cover - exercised with backend/runtime failures
            for utterance in pending_utterances:
                records.append(
                    MetricRecord(
                        run_timestamp_utc=run_timestamp_utc,
                        metric_name="speaker_sim_ecapa",
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
        pending_references.clear()
        pending_num_samples = None

    for wav_path in _iter_wavs_with_progress(model, wav_paths):
        utterance = build_utterance_input(wav_path)
        speaker_id = parse_speaker_id(utterance.utt_id)

        try:
            audio = load_audio_sample(utterance.wav_path, runtime)
            total_audio_sec += audio.duration_sec
            reference_paths = refs_by_speaker.get(speaker_id, [])
            if not reference_paths:
                records.append(
                    MetricRecord(
                        run_timestamp_utc=run_timestamp_utc,
                        metric_name="speaker_sim_ecapa",
                        metric_version=runtime.metric_version,
                        model=model,
                        utt_id=utterance.utt_id,
                        wav_path=str(utterance.wav_path),
                        metric_value=None,
                        status="skip",
                        error=f"missing reference speaker wavs for {speaker_id}",
                    )
                )
                continue

            if speaker_id not in embedding_cache:
                try:
                    embedding_cache[speaker_id] = build_reference_embedding(
                        speaker_id,
                        reference_paths,
                        runtime,
                        batch_size=batch_size,
                    )
                except SkipUtteranceError as exc:
                    embedding_cache[speaker_id] = f"{REFERENCE_EMBEDDING_ERROR_PREFIX}{exc}"

            cached_reference = embedding_cache[speaker_id]
            if isinstance(cached_reference, str) and cached_reference.startswith(REFERENCE_EMBEDDING_ERROR_PREFIX):
                raise SkipUtteranceError(cached_reference.removeprefix(REFERENCE_EMBEDDING_ERROR_PREFIX))
        except SkipUtteranceError as exc:
            records.append(
                MetricRecord(
                    run_timestamp_utc=run_timestamp_utc,
                    metric_name="speaker_sim_ecapa",
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
            records.append(
                MetricRecord(
                    run_timestamp_utc=run_timestamp_utc,
                    metric_name="speaker_sim_ecapa",
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

        current_num_samples = int(audio.waveform.shape[-1])
        should_flush = pending_utterances and pending_num_samples is not None and current_num_samples != pending_num_samples
        if should_flush:
            flush_pending_batch()

        if pending_num_samples is None:
            pending_num_samples = current_num_samples
        pending_utterances.append(utterance)
        pending_audios.append(audio)
        pending_references.append(cached_reference)
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
        "metric_name": "speaker_sim_ecapa",
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
