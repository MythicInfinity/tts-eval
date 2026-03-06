from __future__ import annotations

import math
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
class TokenizedTranscript:
    normalized_text: str
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class CTCScore:
    metric_value: float
    normalized_ctc_loss: float
    decoded_transcript: str


@dataclass(frozen=True)
class CTCRuntime:
    torch: Any
    torchaudio: Any
    bundle: Any
    model: Any
    device: str
    sample_rate: int
    labels: tuple[str, ...]
    metric_version: str


def load_ctc_runtime(device: str) -> CTCRuntime:
    if device != "cuda":
        raise ValueError("CTC runner currently supports only --device cuda")

    try:
        import torch
        import torchaudio
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch and torchaudio must be installed in the runner environment") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required but not available")

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_960H
    labels = tuple(bundle.get_labels())
    model = bundle.get_model().to(device)
    model.eval()
    version = (
        f"torchaudio_{torchaudio.__version__}"
        f"|bundle:WAV2VEC2_ASR_LARGE_960H"
        f"|transform:exp(-ctc_loss/target_len)"
        f"|text:uppercase_space_to_pipe_drop_oov"
    )

    return CTCRuntime(
        torch=torch,
        torchaudio=torchaudio,
        bundle=bundle,
        model=model,
        device=device,
        sample_rate=bundle.sample_rate,
        labels=labels,
        metric_version=version,
    )


def load_audio_sample(path: Path, runtime: CTCRuntime) -> AudioSample:
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


def tokenize_transcript(text: str, labels: tuple[str, ...]) -> TokenizedTranscript:
    vocab = {label: index for index, label in enumerate(labels) if index != 0}
    normalized_chars: list[str] = []
    token_ids: list[int] = []

    for char in text.upper():
        mapped = "|" if char == " " else char
        if mapped in vocab:
            normalized_chars.append(mapped)
            token_ids.append(vocab[mapped])

    normalized_text = "".join(normalized_chars)
    if not token_ids:
        raise SkipUtteranceError("transcript has no valid CTC labels after filtering")

    return TokenizedTranscript(normalized_text=normalized_text, token_ids=tuple(token_ids))


def ctc_closeness_from_loss(total_loss: float, target_length: int) -> tuple[float, float]:
    if target_length <= 0:
        raise ValueError("target_length must be positive")
    normalized_loss = total_loss / target_length
    return math.exp(-normalized_loss), normalized_loss


def decode_greedy(token_ids: list[int], labels: tuple[str, ...], blank_id: int = 0) -> str:
    decoded: list[str] = []
    previous: int | None = None
    for token_id in token_ids:
        if token_id == blank_id:
            previous = None
            continue
        if token_id == previous:
            continue
        decoded.append(labels[token_id])
        previous = token_id
    return "".join(decoded).replace("|", " ")


def score_audio_sample(audio: AudioSample, transcript_text: str, runtime: CTCRuntime) -> CTCScore:
    tokenized = tokenize_transcript(transcript_text, runtime.labels)
    torch = runtime.torch

    with torch.inference_mode():
        waveform = audio.waveform.to(runtime.device)
        emissions, _ = runtime.model(waveform)
        log_probs = torch.nn.functional.log_softmax(emissions, dim=-1).transpose(0, 1)
        targets = torch.tensor(tokenized.token_ids, dtype=torch.long, device=runtime.device)
        input_lengths = torch.full((1,), log_probs.shape[0], dtype=torch.long, device=runtime.device)
        target_lengths = torch.tensor((len(tokenized.token_ids),), dtype=torch.long, device=runtime.device)
        total_loss = torch.nn.functional.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=0,
            reduction="sum",
            zero_infinity=True,
        )
        predicted_ids = emissions.argmax(dim=-1)[0].tolist()

    closeness, normalized_loss = ctc_closeness_from_loss(float(total_loss.item()), len(tokenized.token_ids))
    decoded = decode_greedy(predicted_ids, runtime.labels)
    return CTCScore(metric_value=closeness, normalized_ctc_loss=normalized_loss, decoded_transcript=decoded)


def build_metadata_payload(metric_version: str) -> dict[str, Any]:
    return {
        "metric_name": "ctc_closeness",
        "metric_version": metric_version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "text_policy": "uppercase_space_to_pipe_keep_apostrophe_drop_oov",
        "score_transform": "exp(-ctc_loss/target_len)",
    }


def evaluate_model(model: str, model_dir: Path, runtime: CTCRuntime, run_timestamp_utc: str) -> tuple[list[MetricRecord], float, int]:
    records: list[MetricRecord] = []
    total_audio_sec = 0.0
    n_utts = 0

    for wav_path in iter_wavs(model_dir):
        n_utts += 1
        utterance = build_utterance_input(wav_path)

        try:
            audio = load_audio_sample(utterance.wav_path, runtime)
            total_audio_sec += audio.duration_sec
            if not utterance.txt_path.exists():
                records.append(
                    MetricRecord(
                        run_timestamp_utc=run_timestamp_utc,
                        metric_name="ctc_closeness",
                        metric_version=runtime.metric_version,
                        model=model,
                        utt_id=utterance.utt_id,
                        wav_path=str(utterance.wav_path),
                        metric_value=None,
                        status="skip",
                        error="missing transcript sidecar",
                    )
                )
                continue

            transcript_text = utterance.txt_path.read_text(encoding="utf-8")
            score = score_audio_sample(audio, transcript_text, runtime)
            records.append(
                MetricRecord(
                    run_timestamp_utc=run_timestamp_utc,
                    metric_name="ctc_closeness",
                    metric_version=runtime.metric_version,
                    model=model,
                    utt_id=utterance.utt_id,
                    wav_path=str(utterance.wav_path),
                    metric_value=score.metric_value,
                    status="ok",
                    error=None,
                    decoded_transcript=score.decoded_transcript,
                    normalized_ctc_loss=score.normalized_ctc_loss,
                )
            )
        except SkipUtteranceError as exc:
            records.append(
                MetricRecord(
                    run_timestamp_utc=run_timestamp_utc,
                    metric_name="ctc_closeness",
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
                    metric_name="ctc_closeness",
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
        "metric_name": "ctc_closeness",
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
