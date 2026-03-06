from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class MetricRecord:
    run_timestamp_utc: str
    metric_name: str
    metric_version: str
    model: str
    utt_id: str
    wav_path: str
    metric_value: float | None
    status: str
    error: str | None
    decoded_transcript: str | None = None
    normalized_ctc_loss: float | None = None


def utc_timestamp_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def filename_timestamp(run_timestamp_utc: str) -> str:
    return run_timestamp_utc.replace(":", "-")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[MetricRecord]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=False) + "\n")
