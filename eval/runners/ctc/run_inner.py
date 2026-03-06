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

from tts_eval.ctc import build_metadata_payload, build_summary_payload, evaluate_model, load_ctc_runtime
from tts_eval.discovery import iter_models
from tts_eval.io import filename_timestamp, utc_timestamp_now, write_json, write_jsonl
from tts_eval.progress import log_model_progress, log_model_summary, log_runner_end, log_runner_start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CTC closeness metric over model directories.")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--refs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_timestamp_utc = args.timestamp or utc_timestamp_now()
    timestamp_for_filename = filename_timestamp(run_timestamp_utc)

    runtime = load_ctc_runtime(args.device)
    models = iter_models(args.inputs)
    log_runner_start("ctc", run_timestamp_utc, len(models))

    for index, model_input in enumerate(models, start=1):
        log_model_progress("ctc", model_input.model, index, len(models))
        records, total_audio_sec, n_utts = evaluate_model(
            model=model_input.model,
            model_dir=model_input.model_dir,
            runtime=runtime,
            run_timestamp_utc=run_timestamp_utc,
        )
        model_output_dir = args.output / model_input.model
        summary_payload = build_summary_payload(
            run_timestamp_utc=run_timestamp_utc,
            metric_version=runtime.metric_version,
            model=model_input.model,
            n_utts=n_utts,
            total_audio_sec=total_audio_sec,
            records=records,
        )
        metadata_payload = build_metadata_payload(runtime.metric_version)
        metadata_payload["run_timestamp_utc"] = run_timestamp_utc
        metadata_payload["torch_version"] = runtime.torch.__version__
        metadata_payload["torchaudio_version"] = runtime.torchaudio.__version__
        metadata_payload["bundle"] = "WAV2VEC2_ASR_LARGE_960H"
        metadata_payload["device"] = runtime.device

        write_json(model_output_dir / f"summary_{timestamp_for_filename}.json", summary_payload)
        write_jsonl(model_output_dir / f"per_utt_{timestamp_for_filename}.jsonl", records)
        write_json(model_output_dir / f"metadata_{timestamp_for_filename}.json", metadata_payload)
        log_model_summary("ctc", summary_payload)

    log_runner_end("ctc", len(models))

    return 0


if __name__ == "__main__":
    sys.exit(main())
