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

    def hf_hub_download_compat(*args: Any, **kwargs: Any) -> Any:
        use_auth_token = kwargs.pop("use_auth_token", None)
        if use_auth_token is not None and "token" not in kwargs:
            kwargs["token"] = use_auth_token
        return hf_hub_download(*args, **kwargs)

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


def extract_embedding(audio: AudioSample, runtime: SpeakerSimRuntime) -> Any:
    with runtime.torch.inference_mode():
        embedding = runtime.classifier.encode_batch(audio.waveform.to(runtime.execution_device))

    if hasattr(embedding, "detach"):
        embedding = embedding.detach()
    if hasattr(embedding, "reshape"):
        embedding = embedding.reshape(-1)
    return embedding


def build_reference_embedding(speaker_id: str, reference_paths: list[Path], runtime: SpeakerSimRuntime) -> Any:
    embeddings: list[Any] = []
    for reference_path in reference_paths:
        try:
            audio = load_audio_sample(reference_path, runtime)
        except SkipUtteranceError:
            continue
        embeddings.append(extract_embedding(audio, runtime))

    if not embeddings:
        raise SkipUtteranceError(f"no valid reference wavs for speaker {speaker_id}")

    return runtime.torch.stack(embeddings, dim=0).mean(dim=0)


def score_audio_sample(audio: AudioSample, reference_embedding: Any, runtime: SpeakerSimRuntime) -> float:
    generated_embedding = extract_embedding(audio, runtime)
    similarity = runtime.torch.nn.functional.cosine_similarity(
        generated_embedding.reshape(1, -1),
        reference_embedding.reshape(1, -1),
        dim=-1,
    )
    return float(similarity.item())


def build_metadata_payload(metric_version: str) -> dict[str, Any]:
    return {
        "metric_name": "speaker_sim_ecapa",
        "metric_version": metric_version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "checkpoint": SPEAKER_SIM_CHECKPOINT,
        "reference_pooling": "mean_embedding",
        "similarity": "cosine_similarity",
    }


def evaluate_model(
    model: str,
    model_dir: Path,
    refs_dir: Path,
    runtime: SpeakerSimRuntime,
    run_timestamp_utc: str,
    reference_wavs_by_speaker: dict[str, list[Path]] | None = None,
    reference_embedding_cache: dict[str, Any] | None = None,
) -> tuple[list[MetricRecord], float, int]:
    records: list[MetricRecord] = []
    total_audio_sec = 0.0
    n_utts = 0
    refs_by_speaker = reference_wavs_by_speaker or index_reference_wavs(refs_dir)
    embedding_cache = reference_embedding_cache if reference_embedding_cache is not None else {}

    for wav_path in iter_wavs(model_dir):
        n_utts += 1
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
                    embedding_cache[speaker_id] = build_reference_embedding(speaker_id, reference_paths, runtime)
                except SkipUtteranceError as exc:
                    embedding_cache[speaker_id] = f"{REFERENCE_EMBEDDING_ERROR_PREFIX}{exc}"

            cached_reference = embedding_cache[speaker_id]
            if isinstance(cached_reference, str) and cached_reference.startswith(REFERENCE_EMBEDDING_ERROR_PREFIX):
                raise SkipUtteranceError(cached_reference.removeprefix(REFERENCE_EMBEDDING_ERROR_PREFIX))

            similarity = score_audio_sample(audio, cached_reference, runtime)
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
