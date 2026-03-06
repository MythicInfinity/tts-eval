from __future__ import annotations

import argparse
import os
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
from tts_eval.io import filename_timestamp, utc_timestamp_now, write_json, write_jsonl
from tts_eval.progress import log_model_progress, log_model_summary, log_runner_end, log_runner_start
from tts_eval.utmos import build_metadata_payload, build_summary_payload, evaluate_model, load_utmos_runtime


def _default_num_workers() -> int:
    return max(1, min(8, os.cpu_count() or 1))


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UTMOS over model directories.")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--refs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=_default_num_workers())
    parser.add_argument("--remove-silent-section", type=_parse_bool, default=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_timestamp_utc = args.timestamp or utc_timestamp_now()
    timestamp_for_filename = filename_timestamp(run_timestamp_utc)
    runtime = load_utmos_runtime(
        execution_device=args.device,
        remove_silent_section=args.remove_silent_section,
    )
    models = iter_models(args.inputs)
    log_runner_start("utmos", run_timestamp_utc, len(models))

    for index, model_input in enumerate(models, start=1):
        log_model_progress("utmos", model_input.model, index, len(models))
        records, total_audio_sec, n_utts = evaluate_model(
            model=model_input.model,
            model_dir=model_input.model_dir,
            runtime=runtime,
            run_timestamp_utc=run_timestamp_utc,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
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
        metadata_payload = build_metadata_payload(
            runtime.metric_version,
            config=runtime.config,
            fold=runtime.fold,
            seed=runtime.seed,
            remove_silent_section=runtime.remove_silent_section,
            predict_dataset=runtime.predict_dataset,
            num_repetitions=runtime.num_repetitions,
        )
        metadata_payload["run_timestamp_utc"] = run_timestamp_utc
        metadata_payload["torch_version"] = runtime.torch.__version__
        metadata_payload["torchaudio_version"] = runtime.torchaudio.__version__
        metadata_payload["utmosv2_version"] = runtime.utmosv2.__version__
        metadata_payload["execution_device"] = runtime.execution_device
        metadata_payload["batch_size"] = args.batch_size
        metadata_payload["num_workers"] = args.num_workers
        metadata_payload["remove_silent_section"] = args.remove_silent_section
        write_json(model_output_dir / f"summary_{timestamp_for_filename}.json", summary_payload)
        write_jsonl(model_output_dir / f"per_utt_{timestamp_for_filename}.jsonl", records)
        write_json(model_output_dir / f"metadata_{timestamp_for_filename}.json", metadata_payload)
        log_model_summary("utmos", summary_payload)

    log_runner_end("utmos", len(models))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
