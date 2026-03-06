from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


SUMMARY_RE = re.compile(r"^summary_(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z)\.json$")


def parse_summary_timestamp(path: Path) -> str | None:
    match = SUMMARY_RE.match(path.name)
    if not match:
        return None
    return match.group("timestamp")


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_latest_summaries(eval_root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    latest: dict[str, dict[str, tuple[str, Path]]] = defaultdict(dict)

    if not eval_root.exists():
        return {}

    for metric_dir in sorted(path for path in eval_root.iterdir() if path.is_dir()):
        metric_name = metric_dir.name
        for model_dir in sorted(path for path in metric_dir.iterdir() if path.is_dir()):
            model = model_dir.name
            for summary_path in sorted(model_dir.glob("summary_*.json")):
                timestamp = parse_summary_timestamp(summary_path)
                if timestamp is None:
                    continue
                previous = latest[metric_name].get(model)
                if previous is None or timestamp > previous[0]:
                    latest[metric_name][model] = (timestamp, summary_path)

    return {
        metric_name: {model: load_summary(path) for model, (_, path) in models.items()}
        for metric_name, models in latest.items()
    }


def build_coalesced_rows(latest_summaries: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    models = sorted({model for models in latest_summaries.values() for model in models})
    rows: list[dict[str, Any]] = []

    for model in models:
        ctc_summary = latest_summaries.get("ctc", {}).get(model)
        if ctc_summary is None:
            continue

        rows.append(
            {
                "run_timestamp_utc": ctc_summary["run_timestamp_utc"],
                "model": model,
                "n_utts": ctc_summary["n_utts"],
                "total_audio_sec": ctc_summary["total_audio_sec"],
                "ctc_closeness_mean": ctc_summary["metric_mean"],
            }
        )

    return rows
