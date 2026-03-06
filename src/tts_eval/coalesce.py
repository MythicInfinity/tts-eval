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

    for metric_dir in _iter_metric_dirs(eval_root):
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


def _iter_metric_dirs(eval_root: Path) -> list[Path]:
    metric_dirs: dict[Path, None] = {}

    direct_metrics_root = eval_root / "data" / "evals"
    if direct_metrics_root.exists():
        for metric_dir in sorted(path for path in direct_metrics_root.iterdir() if path.is_dir()):
            metric_dirs[metric_dir] = None

    for metric_dir in sorted(path for path in eval_root.iterdir() if path.is_dir()):
        if any(child.is_dir() for child in metric_dir.iterdir()):
            has_summary = any(metric_dir.glob("*/*summary_*.json")) or any(metric_dir.glob("*/summary_*.json"))
            if has_summary:
                metric_dirs[metric_dir] = None

    for nested in sorted(eval_root.rglob("data/evals/*")):
        if nested.is_dir():
            metric_dirs[nested] = None

    return sorted(metric_dirs)


def build_coalesced_rows(latest_summaries: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    models = sorted({model for models in latest_summaries.values() for model in models})
    rows: list[dict[str, Any]] = []

    for model in models:
        ctc_summary = latest_summaries.get("ctc", {}).get(model)
        ttsds2_summary = latest_summaries.get("ttsds2", {}).get(model)
        dnsmos_summary = latest_summaries.get("dnsmos", {}).get(model)
        speaker_sim_summary = latest_summaries.get("speaker_sim", {}).get(model)

        if ctc_summary is None and ttsds2_summary is None and dnsmos_summary is None and speaker_sim_summary is None:
            continue

        base_summary = ctc_summary or dnsmos_summary or speaker_sim_summary or ttsds2_summary
        rows.append(
            {
                "run_timestamp_utc": _max_timestamp(ctc_summary, ttsds2_summary, dnsmos_summary, speaker_sim_summary),
                "model": model,
                "n_utts": base_summary["n_utts"],
                "total_audio_sec": base_summary["total_audio_sec"],
                "ctc_closeness_mean": ctc_summary["metric_mean"] if ctc_summary else None,
                "ttsds2_total": ttsds2_summary["metric_value"] if ttsds2_summary else None,
                "dnsmos_ovrl_mean": dnsmos_summary["metric_mean"] if dnsmos_summary else None,
                "speaker_sim_ecapa_mean": speaker_sim_summary["metric_mean"] if speaker_sim_summary else None,
            }
        )

    return rows


def _max_timestamp(*summaries: dict[str, Any] | None) -> str | None:
    timestamps = [summary["run_timestamp_utc"] for summary in summaries if summary is not None]
    if not timestamps:
        return None
    return max(timestamps)
