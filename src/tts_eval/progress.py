from __future__ import annotations

from typing import Any


def _print(message: str) -> None:
    print(message, flush=True)


def log_runner_start(metric_name: str, run_timestamp_utc: str, model_count: int) -> None:
    _print(f"[{metric_name}] start run_timestamp_utc={run_timestamp_utc} models={model_count}")


def log_model_progress(metric_name: str, model: str, index: int, total: int) -> None:
    _print(f"[{metric_name}] progress model={model} index={index}/{total}")


def log_model_summary(metric_name: str, summary_payload: dict[str, Any]) -> None:
    model = summary_payload["model"]
    n_utts = summary_payload.get("n_utts")
    total_audio_sec = summary_payload.get("total_audio_sec")
    fail_count = summary_payload.get("fail_count")
    skip_count = summary_payload.get("skip_count")
    metric_value = summary_payload.get("metric_mean")
    if metric_value is None:
        metric_value = summary_payload.get("metric_value")
    ce_mean = summary_payload.get("ce_mean")
    pq_mean = summary_payload.get("pq_mean")
    error = summary_payload.get("error")
    if metric_value is None and isinstance(ce_mean, (int, float)) and isinstance(pq_mean, (int, float)):
        metric_display = f"ce={ce_mean:.6f},pq={pq_mean:.6f}"
    else:
        metric_display = "null" if metric_value is None else f"{metric_value:.6f}"

    parts = [
        f"[{metric_name}] done",
        f"model={model}",
        f"score={metric_display}",
        f"n_utts={n_utts}",
        f"total_audio_sec={total_audio_sec}",
        f"skip_count={skip_count}",
        f"fail_count={fail_count}",
    ]
    if error:
        parts.append(f"error={error}")
    _print(" ".join(parts))


def log_runner_end(metric_name: str, model_count: int) -> None:
    _print(f"[{metric_name}] finished models={model_count}")
