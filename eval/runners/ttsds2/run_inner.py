from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
SRC_CANDIDATES = [
    THIS_FILE.parents[3] / "src",
    THIS_FILE.parent / "src",
]
for src_dir in SRC_CANDIDATES:
    if src_dir.exists():
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        break

from tts_eval.discovery import iter_models
from tts_eval.io import filename_timestamp, utc_timestamp_now, write_json
from tts_eval.ttsds2 import evaluate_model, load_ttsds2_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TTSDS2 metric over model directories.")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--refs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timestamp", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_timestamp_utc = args.timestamp or utc_timestamp_now()
    timestamp_for_filename = filename_timestamp(run_timestamp_utc)
    runtime = load_ttsds2_runtime()

    for model_input in iter_models(args.inputs):
        summary_payload, metadata_payload = evaluate_model(
            model=model_input.model,
            model_dir=model_input.model_dir,
            refs_dir=args.refs,
            runtime=runtime,
            run_timestamp_utc=run_timestamp_utc,
        )
        model_output_dir = args.output / model_input.model
        write_json(model_output_dir / f"summary_{timestamp_for_filename}.json", summary_payload)
        write_json(model_output_dir / f"metadata_{timestamp_for_filename}.json", metadata_payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
