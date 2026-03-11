from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tts_eval.coalesce import build_coalesced_rows, collect_latest_summaries


class CoalesceTests(unittest.TestCase):
    def test_collect_latest_summaries_finds_runner_local_eval_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            model_dir = repo_root / "eval" / "runners" / "dnsmos" / "data" / "evals" / "dnsmos" / "model_a"
            model_dir.mkdir(parents=True)
            summary_path = model_dir / "summary_2026-03-06T02-00-00Z.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "run_timestamp_utc": "2026-03-06T02:00:00Z",
                        "model": "model_a",
                        "n_utts": 2,
                        "total_audio_sec": 2.0,
                        "metric_mean": 3.1,
                    }
                ),
                encoding="utf-8",
            )

            latest = collect_latest_summaries(repo_root)

        self.assertEqual(latest["dnsmos"]["model_a"]["metric_mean"], 3.1)

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
                    "ctc_tortoise_closeness_mean": None,
                    "dnsmos_ovrl_mean": None,
                    "nisqa_mos_mean": None,
                    "speaker_sim_ecapa_mean": None,
                    "utmos_mean": None,
                    "audiobox_ce_mean": None,
                    "audiobox_pq_mean": None,
                }
            ],
        )

    def test_build_coalesced_rows_merges_tortoise_ctc(self) -> None:
        rows = build_coalesced_rows(
            {
                "ctc_tortoise": {
                    "model_a": {
                        "run_timestamp_utc": "2026-03-06T03:00:00Z",
                        "model": "model_a",
                        "n_utts": 2,
                        "total_audio_sec": 2.0,
                        "metric_mean": 0.81,
                    }
                }
            }
        )
        self.assertEqual(
            rows,
            [
                {
                    "run_timestamp_utc": "2026-03-06T03:00:00Z",
                    "model": "model_a",
                    "n_utts": 2,
                    "total_audio_sec": 2.0,
                    "ctc_closeness_mean": None,
                    "ctc_tortoise_closeness_mean": 0.81,
                    "dnsmos_ovrl_mean": None,
                    "nisqa_mos_mean": None,
                    "speaker_sim_ecapa_mean": None,
                    "utmos_mean": None,
                    "audiobox_ce_mean": None,
                    "audiobox_pq_mean": None,
                }
            ],
        )

    def test_build_coalesced_rows_merges_dnsmos(self) -> None:
        rows = build_coalesced_rows(
            {
                "dnsmos": {
                    "model_a": {
                        "run_timestamp_utc": "2026-03-06T04:00:00Z",
                        "model": "model_a",
                        "n_utts": 3,
                        "total_audio_sec": 5.0,
                        "metric_mean": 3.2,
                    }
                }
            }
        )
        self.assertEqual(
            rows,
            [
                {
                    "run_timestamp_utc": "2026-03-06T04:00:00Z",
                    "model": "model_a",
                    "n_utts": 3,
                    "total_audio_sec": 5.0,
                    "ctc_closeness_mean": None,
                    "ctc_tortoise_closeness_mean": None,
                    "dnsmos_ovrl_mean": 3.2,
                    "nisqa_mos_mean": None,
                    "speaker_sim_ecapa_mean": None,
                    "utmos_mean": None,
                    "audiobox_ce_mean": None,
                    "audiobox_pq_mean": None,
                }
            ],
        )

    def test_build_coalesced_rows_merges_nisqa(self) -> None:
        rows = build_coalesced_rows(
            {
                "nisqa": {
                    "model_a": {
                        "run_timestamp_utc": "2026-03-06T04:30:00Z",
                        "model": "model_a",
                        "n_utts": 3,
                        "total_audio_sec": 5.0,
                        "metric_mean": 3.8,
                    }
                }
            }
        )
        self.assertEqual(
            rows,
            [
                {
                    "run_timestamp_utc": "2026-03-06T04:30:00Z",
                    "model": "model_a",
                    "n_utts": 3,
                    "total_audio_sec": 5.0,
                    "ctc_closeness_mean": None,
                    "ctc_tortoise_closeness_mean": None,
                    "dnsmos_ovrl_mean": None,
                    "nisqa_mos_mean": 3.8,
                    "speaker_sim_ecapa_mean": None,
                    "utmos_mean": None,
                    "audiobox_ce_mean": None,
                    "audiobox_pq_mean": None,
                }
            ],
        )

    def test_build_coalesced_rows_merges_speaker_similarity(self) -> None:
        rows = build_coalesced_rows(
            {
                "speaker_sim": {
                    "model_a": {
                        "run_timestamp_utc": "2026-03-06T05:00:00Z",
                        "model": "model_a",
                        "n_utts": 4,
                        "total_audio_sec": 6.0,
                        "metric_mean": 0.92,
                    }
                }
            }
        )
        self.assertEqual(
            rows,
            [
                {
                    "run_timestamp_utc": "2026-03-06T05:00:00Z",
                    "model": "model_a",
                    "n_utts": 4,
                    "total_audio_sec": 6.0,
                    "ctc_closeness_mean": None,
                    "ctc_tortoise_closeness_mean": None,
                    "dnsmos_ovrl_mean": None,
                    "nisqa_mos_mean": None,
                    "speaker_sim_ecapa_mean": 0.92,
                    "utmos_mean": None,
                    "audiobox_ce_mean": None,
                    "audiobox_pq_mean": None,
                }
            ],
        )

    def test_build_coalesced_rows_merges_utmos(self) -> None:
        rows = build_coalesced_rows(
            {
                "utmos": {
                    "model_a": {
                        "run_timestamp_utc": "2026-03-06T06:00:00Z",
                        "model": "model_a",
                        "n_utts": 5,
                        "total_audio_sec": 7.5,
                        "metric_mean": 4.15,
                    }
                }
            }
        )
        self.assertEqual(
            rows,
            [
                {
                    "run_timestamp_utc": "2026-03-06T06:00:00Z",
                    "model": "model_a",
                    "n_utts": 5,
                    "total_audio_sec": 7.5,
                    "ctc_closeness_mean": None,
                    "ctc_tortoise_closeness_mean": None,
                    "dnsmos_ovrl_mean": None,
                    "nisqa_mos_mean": None,
                    "speaker_sim_ecapa_mean": None,
                    "utmos_mean": 4.15,
                    "audiobox_ce_mean": None,
                    "audiobox_pq_mean": None,
                }
            ],
        )

    def test_build_coalesced_rows_merges_audiobox(self) -> None:
        rows = build_coalesced_rows(
            {
                "audiobox": {
                    "model_a": {
                        "run_timestamp_utc": "2026-03-06T06:30:00Z",
                        "model": "model_a",
                        "n_utts": 5,
                        "total_audio_sec": 7.5,
                        "ce_mean": 5.11,
                        "pq_mean": 6.02,
                    }
                }
            }
        )
        self.assertEqual(
            rows,
            [
                {
                    "run_timestamp_utc": "2026-03-06T06:30:00Z",
                    "model": "model_a",
                    "n_utts": 5,
                    "total_audio_sec": 7.5,
                    "ctc_closeness_mean": None,
                    "ctc_tortoise_closeness_mean": None,
                    "dnsmos_ovrl_mean": None,
                    "nisqa_mos_mean": None,
                    "speaker_sim_ecapa_mean": None,
                    "utmos_mean": None,
                    "audiobox_ce_mean": 5.11,
                    "audiobox_pq_mean": 6.02,
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
