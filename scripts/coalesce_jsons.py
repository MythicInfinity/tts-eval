from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tts_eval.coalesce import build_coalesced_rows, collect_latest_summaries
from tts_eval.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coalesce latest metric summaries into one JSON file.")
    parser.add_argument("--eval-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("data/evals/coalesced_summary.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_coalesced_rows(collect_latest_summaries(args.eval_root))
    write_json(args.output, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
