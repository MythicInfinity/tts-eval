from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Iterable

from tts_eval.io import MetricRecord


@dataclass(frozen=True)
class AggregateStats:
    metric_mean: float | None
    metric_median: float | None
    metric_std: float | None
    ok_count: int
    fail_count: int
    skip_count: int


def aggregate_metric_records(records: Iterable[MetricRecord]) -> AggregateStats:
    materialized = list(records)
    values = [record.metric_value for record in materialized if record.status == "ok" and record.metric_value is not None]
    ok_count = len(values)
    fail_count = sum(record.status == "fail" for record in materialized)
    skip_count = sum(record.status == "skip" for record in materialized)

    if not values:
        return AggregateStats(
            metric_mean=None,
            metric_median=None,
            metric_std=None,
            ok_count=ok_count,
            fail_count=fail_count,
            skip_count=skip_count,
        )

    metric_mean = statistics.fmean(values)
    metric_median = statistics.median(values)
    metric_std = statistics.stdev(values) if len(values) > 1 else 0.0

    if math.isnan(metric_mean) or math.isnan(metric_median) or math.isnan(metric_std):
        raise ValueError("aggregate statistics contain NaN")

    return AggregateStats(
        metric_mean=metric_mean,
        metric_median=metric_median,
        metric_std=metric_std,
        ok_count=ok_count,
        fail_count=fail_count,
        skip_count=skip_count,
    )
