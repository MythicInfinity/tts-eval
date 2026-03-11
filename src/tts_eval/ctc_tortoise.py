from __future__ import annotations

import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tts_eval.ctc import AudioSample, CTCScore, SkipUtteranceError, ctc_closeness_from_loss
from tts_eval.discovery import build_utterance_input, iter_wavs
from tts_eval.io import MetricRecord
from tts_eval.stats import aggregate_metric_records

_PAD = "_"
_SPECIAL = "-"
_PUNCTUATION = "!'(),.:;? "
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_SYMBOLS = (_PAD,) + tuple(_SPECIAL) + tuple(_PUNCTUATION) + tuple(_LETTERS)
_SYMBOL_TO_ID = {symbol: index for index, symbol in enumerate(_SYMBOLS)}
_WHITESPACE_RE = re.compile(r"\s+")
_CURLY_RE = re.compile(r"(.*?)\{(.+?)\}(.*)")
_ABBREVIATIONS = tuple(
    (re.compile(rf"\b{source}\.", re.IGNORECASE), replacement)
    for source, replacement in (
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    )
)
_COMMA_NUMBER_RE = re.compile(r"([0-9][0-9,]+[0-9])")
_DECIMAL_NUMBER_RE = re.compile(r"([0-9]+\.[0-9]+)")
_POUNDS_RE = re.compile(r"£([0-9,]*[0-9]+)")
_DOLLARS_RE = re.compile(r"\$([0-9\.,]*[0-9]+)")
_ORDINAL_RE = re.compile(r"[0-9]+(st|nd|rd|th)")
_NUMBER_RE = re.compile(r"[0-9]+")

try:
    import inflect
except ModuleNotFoundError:  # pragma: no cover - runner image installs inflect
    inflect = None

try:
    from unidecode import unidecode as _unidecode
except ModuleNotFoundError:  # pragma: no cover - runner image installs Unidecode
    def _unidecode(value: str) -> str:
        return value


@dataclass(frozen=True)
class TokenizedTranscript:
    normalized_text: str
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class TortoiseCTCRuntime:
    torch: Any
    torchaudio: Any
    transformers: Any
    model: Any
    tokenizer: Any
    vocab: dict[str, int]
    device: str
    sample_rate: int
    labels: tuple[str, ...]
    metric_version: str
    model_id: str
    vocab_repo_id: str
    blank_id: int
    ctc_zero_infinity: bool
    model_vocab_size: int


def _make_inflect_engine() -> Any | None:
    if inflect is None:
        return None
    return inflect.engine()


_INFLECT = _make_inflect_engine()


def _remove_commas(match: re.Match[str]) -> str:
    return match.group(1).replace(",", "")


def _expand_decimal_point(match: re.Match[str]) -> str:
    return match.group(1).replace(".", " point ")


def _number_to_words(number: int, *, ordinal: bool = False) -> str:
    if _INFLECT is None:
        return str(number)
    if ordinal:
        return _INFLECT.number_to_words(f"{number}{_ordinal_suffix(number)}")
    if number > 1000 and number < 3000:
        if number == 2000:
            return "two thousand"
        if number > 2000 and number < 2010:
            return "two thousand " + _INFLECT.number_to_words(number % 100)
        if number % 100 == 0:
            return _INFLECT.number_to_words(number // 100) + " hundred"
        return _INFLECT.number_to_words(number, andword="", zero="oh", group=2).replace(", ", " ")
    return _INFLECT.number_to_words(number, andword="")


def _ordinal_suffix(number: int) -> str:
    if 10 <= number % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")


def _expand_dollars(match: re.Match[str]) -> str:
    value = match.group(1)
    parts = value.split(".")
    if len(parts) > 2:
        return value + " dollars"
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        cent_unit = "cent" if cents == 1 else "cents"
        return f"{dollars} {dollar_unit}, {cents} {cent_unit}"
    if dollars:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        return f"{dollars} {dollar_unit}"
    if cents:
        cent_unit = "cent" if cents == 1 else "cents"
        return f"{cents} {cent_unit}"
    return "zero dollars"


def _expand_ordinal(match: re.Match[str]) -> str:
    value = match.group(0)
    number = int(value[:-2])
    return _number_to_words(number, ordinal=True)


def _expand_number(match: re.Match[str]) -> str:
    return _number_to_words(int(match.group(0)))


def normalize_numbers(text: str) -> str:
    text = re.sub(_COMMA_NUMBER_RE, _remove_commas, text)
    text = re.sub(_POUNDS_RE, r"\1 pounds", text)
    text = re.sub(_DOLLARS_RE, _expand_dollars, text)
    text = re.sub(_DECIMAL_NUMBER_RE, _expand_decimal_point, text)
    text = re.sub(_ORDINAL_RE, _expand_ordinal, text)
    return re.sub(_NUMBER_RE, _expand_number, text)


def english_cleaners(text: str) -> str:
    text = _unidecode(text)
    text = text.lower()
    text = normalize_numbers(text)
    for regex, replacement in _ABBREVIATIONS:
        text = re.sub(regex, replacement, text)
    text = re.sub(_WHITESPACE_RE, " ", text)
    return text.replace('"', "")


def _lookup_vocab_token(symbol: str, symbol_to_id: dict[str, int]) -> int | None:
    token = "|" if symbol == " " and "|" in symbol_to_id else symbol
    token_id = symbol_to_id.get(token)
    if token_id is None:
        return None
    if token in {_PAD, "~", "<pad>"}:
        return None
    return token_id


def _symbols_to_sequence(symbols: str, symbol_to_id: dict[str, int]) -> list[int]:
    sequence: list[int] = []
    for symbol in symbols:
        token_id = _lookup_vocab_token(symbol, symbol_to_id)
        if token_id is not None:
            sequence.append(token_id)
    return sequence


def _arpabet_to_sequence(text: str, symbol_to_id: dict[str, int]) -> list[int]:
    sequence: list[int] = []
    for symbol in text.split():
        token_id = symbol_to_id.get(f"@{symbol}")
        if token_id is not None:
            sequence.append(token_id)
    return sequence


def text_to_sequence(text: str, symbol_to_id: dict[str, int] | None = None) -> list[int]:
    active_symbol_to_id = symbol_to_id or _SYMBOL_TO_ID
    sequence: list[int] = []
    remaining = text
    while remaining:
        match = _CURLY_RE.match(remaining)
        if not match:
            sequence.extend(_symbols_to_sequence(english_cleaners(remaining), active_symbol_to_id))
            break
        sequence.extend(_symbols_to_sequence(english_cleaners(match.group(1)), active_symbol_to_id))
        sequence.extend(_arpabet_to_sequence(match.group(2), active_symbol_to_id))
        remaining = match.group(3)
    return sequence


def build_vocab_labels(vocab: dict[str, int], model_vocab_size: int) -> tuple[str, ...]:
    labels = [""] * model_vocab_size
    for token, token_id in vocab.items():
        if token_id < 0 or token_id >= model_vocab_size:
            continue
        if token == "|":
            labels[token_id] = " "
            continue
        labels[token_id] = token
    return tuple(labels)


def _vocab_token_aliases(symbol: str) -> tuple[str, ...]:
    if symbol == " ":
        return ("|",)
    if symbol == _PAD:
        return (_PAD, "<pad>")
    return (symbol,)


def validate_vocab_alignment(vocab: dict[str, int]) -> None:
    for symbol, expected_id in _SYMBOL_TO_ID.items():
        vocab_tokens = _vocab_token_aliases(symbol)
        actual_id = next((vocab[token] for token in vocab_tokens if token in vocab), None)
        if actual_id != expected_id:
            display_token = vocab_tokens[0]
            raise RuntimeError(
                f"unexpected tortoise CTC vocab mapping for {display_token!r}: "
                f"expected id {expected_id}, got {actual_id}"
            )


def load_tortoise_ctc_runtime(
    device: str,
    model_id: str = "jbetker/wav2vec2-large-robust-ft-libritts-voxpopuli",
    vocab_repo_id: str = "jbetker/tacotron_symbols",
) -> TortoiseCTCRuntime:
    try:
        import torch
        import torchaudio
        import transformers
        from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch, torchaudio, and transformers must be installed in the runner environment") from exc

    if not device.startswith("cuda"):
        raise ValueError("Tortoise CTC runner currently supports only CUDA devices")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required but not available")

    model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)
    model.eval()
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(vocab_repo_id, do_lower_case=False)
    vocab = tokenizer.get_vocab()
    validate_vocab_alignment(vocab)
    labels = build_vocab_labels(vocab, model.config.vocab_size)
    metric_version = (
        f"transformers_{transformers.__version__}"
        f"|model:{model_id}"
        f"|vocab:{vocab_repo_id}"
        f"|transform:exp(-ctc_loss/target_len)"
        f"|text:tortoise_english_cleaners"
    )

    return TortoiseCTCRuntime(
        torch=torch,
        torchaudio=torchaudio,
        transformers=transformers,
        model=model,
        tokenizer=tokenizer,
        vocab=vocab,
        device=device,
        sample_rate=16000,
        labels=labels,
        metric_version=metric_version,
        model_id=model_id,
        vocab_repo_id=vocab_repo_id,
        blank_id=model.config.pad_token_id,
        ctc_zero_infinity=bool(model.config.ctc_zero_infinity),
        model_vocab_size=model.config.vocab_size,
    )


def load_audio_sample(path: Path, runtime: TortoiseCTCRuntime) -> AudioSample:
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


def tokenize_transcript(
    text: str,
    symbol_to_id: dict[str, int] | None = None,
    tokenizer: Any | None = None,
    model_vocab_size: int | None = None,
) -> TokenizedTranscript:
    if tokenizer is not None:
        cleaned_text = english_cleaners(text)
        encoded = tokenizer(cleaned_text, add_special_tokens=False)
        normalized_ids = [
            token_id
            for token_id in encoded.input_ids
            if isinstance(token_id, int)
            and token_id >= 0
            and (model_vocab_size is None or token_id < model_vocab_size)
        ]
        normalized_text = cleaned_text
    else:
        normalized_ids = text_to_sequence(text, symbol_to_id=symbol_to_id)
        normalized_text = english_cleaners(text)
    if not normalized_ids:
        raise SkipUtteranceError("transcript has no valid tortoise CTC labels after filtering")
    return TokenizedTranscript(normalized_text=normalized_text, token_ids=tuple(normalized_ids))


def normalize_waveform_for_model(waveform: Any, torch_module: Any) -> Any:
    mean = waveform.mean()
    variance = waveform.var()
    return ((waveform - mean) / torch_module.sqrt(variance + 1e-7)).squeeze(1)


def decode_greedy(
    token_ids: list[int],
    labels: tuple[str, ...],
    blank_id: int,
    skip_token_ids: set[int] | None = None,
) -> str:
    decoded: list[str] = []
    previous: int | None = None
    skip_ids = skip_token_ids or set()
    for token_id in token_ids:
        if token_id == blank_id:
            previous = None
            continue
        if token_id == previous:
            continue
        if token_id in skip_ids:
            previous = token_id
            continue
        if token_id < 0 or token_id >= len(labels):
            previous = token_id
            continue
        decoded.append(labels[token_id])
        previous = token_id
    return "".join(decoded)


def score_audio_sample(audio: AudioSample, transcript_text: str, runtime: TortoiseCTCRuntime) -> CTCScore:
    tokenized = tokenize_transcript(
        transcript_text,
        tokenizer=runtime.tokenizer,
        model_vocab_size=runtime.model_vocab_size,
    )
    torch = runtime.torch

    with torch.inference_mode():
        waveform = audio.waveform.to(runtime.device)
        normalized_waveform = normalize_waveform_for_model(waveform, torch)
        targets = torch.tensor(tokenized.token_ids, dtype=torch.long, device=runtime.device).unsqueeze(0)
        outputs = runtime.model(input_values=normalized_waveform, labels=targets)
        total_loss = outputs.loss
        predicted_ids = outputs.logits.argmax(dim=-1)[0].tolist()

    closeness, normalized_loss = ctc_closeness_from_loss(float(total_loss.item()), len(tokenized.token_ids))
    decoded = decode_greedy(
        predicted_ids,
        runtime.labels,
        blank_id=runtime.blank_id,
    )
    return CTCScore(metric_value=closeness, normalized_ctc_loss=normalized_loss, decoded_transcript=decoded)


def build_metadata_payload(metric_version: str) -> dict[str, Any]:
    return {
        "metric_name": "ctc_tortoise_closeness",
        "metric_version": metric_version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "text_policy": "tortoise_english_cleaners",
        "tokenizer_source": "tts_scores.tokenizer.text_to_sequence",
        "score_transform": "exp(-ctc_loss/target_len)",
    }


def evaluate_model(model: str, model_dir: Path, runtime: TortoiseCTCRuntime, run_timestamp_utc: str) -> tuple[list[MetricRecord], float, int]:
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
                        metric_name="ctc_tortoise_closeness",
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
                    metric_name="ctc_tortoise_closeness",
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
                    metric_name="ctc_tortoise_closeness",
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
                    metric_name="ctc_tortoise_closeness",
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
        "metric_name": "ctc_tortoise_closeness",
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
