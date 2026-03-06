from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tts_eval.coalesce import build_coalesced_rows, collect_latest_summaries


class CoalesceTests(unittest.TestCase):
    def test_collect_latest_summaries_prefers_newest_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            eval_root = Path(tmpdir)
            model_dir = eval_root / "ctc" / "model_a"
            model_dir.mkdir(parents=True)

            older = model_dir / "summary_2026-03-06T01-00-00Z.json"
            newer = model_dir / "summary_2026-03-06T02-00-00Z.json"
            older.write_text(
                json.dumps({"run_timestamp_utc": "2026-03-06T01:00:00Z", "model": "model_a", "n_utts": 1, "total_audio_sec": 1.0, "metric_mean": 0.5}),
                encoding="utf-8",
            )
            newer.write_text(
                json.dumps({"run_timestamp_utc": "2026-03-06T02:00:00Z", "model": "model_a", "n_utts": 2, "total_audio_sec": 2.0, "metric_mean": 0.7}),
                encoding="utf-8",
            )

            latest = collect_latest_summaries(eval_root)
            self.assertEqual(latest["ctc"]["model_a"]["metric_mean"], 0.7)

    def test_build_coalesced_rows_emits_ctc_shape(self) -> None:
        rows = build_coalesced_rows(
            {
                "ctc": {
                    "model_a": {
                        "run_timestamp_utc": "2026-03-06T02:00:00Z",
                        "model": "model_a",
                        "n_utts": 2,
                        "total_audio_sec": 2.0,
                        "metric_mean": 0.7,
                    }
                }
            }
        )
        self.assertEqual(
            rows,
            [
                {
                    "run_timestamp_utc": "2026-03-06T02:00:00Z",
                    "model": "model_a",
                    "n_utts": 2,
                    "total_audio_sec": 2.0,
                    "ctc_closeness_mean": 0.7,
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
