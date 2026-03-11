from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
SRC_CANDIDATES = [
    THIS_FILE.parents[2] / "src",
    THIS_FILE.parent / "src",
]
for src_dir in SRC_CANDIDATES:
    if src_dir.exists():
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        break

from tts_eval.chatterbox_turbo import (
    load_chatterbox_turbo_runtime,
    synthesize_request,
    validate_reference_wavs,
)
from tts_eval.io import utc_timestamp_now
from tts_eval.model_runner_inputs import (
    TARGET_UTTERANCES_PER_SPEAKER,
    build_generation_requests,
    count_generation_requests,
    index_speaker_references,
)
from tts_eval.progress import log_generation_summary, log_model_progress, log_runner_end, log_runner_start
from tts_eval.utterance_dataset_config import DEFAULT_UTTERANCE_TEXT_DATASET_SPECS
from tts_eval.utterance_texts import build_utterance_text_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate eval-ready WAV/TXT pairs with Chatterbox Turbo.")
    parser.add_argument("--refs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    return parser.parse_args()


def _build_staging_dir(final_output_dir: Path) -> Path:
    parent_dir = final_output_dir.parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    staging_path = Path(tempfile.mkdtemp(prefix=f".{final_output_dir.name}_", dir=parent_dir))
    return staging_path


def main() -> int:
    args = parse_args()
    run_timestamp_utc = args.timestamp or utc_timestamp_now()
    model_name = args.output.name

    utterance_texts = build_utterance_text_dataset(DEFAULT_UTTERANCE_TEXT_DATASET_SPECS)
    speaker_references = validate_reference_wavs(index_speaker_references(args.refs))
    total_requests = count_generation_requests(
        speaker_references,
        target_utterances_per_speaker=TARGET_UTTERANCES_PER_SPEAKER,
    )
    runtime = load_chatterbox_turbo_runtime(args.device)

    log_runner_start(model_name, run_timestamp_utc, 1)
    conditionals_cache: dict[Path, object] = {}
    generated_count = 0
    staging_output_dir: Path | None = None

    try:
        staging_output_dir = _build_staging_dir(args.output)
        requests = build_generation_requests(
            speaker_references,
            utterance_texts,
            staging_output_dir,
            target_utterances_per_speaker=TARGET_UTTERANCES_PER_SPEAKER,
        )
        for index, request in enumerate(requests, start=1):
            log_model_progress(model_name, request.utterance_id, index, total_requests)
            synthesize_request(
                request,
                runtime,
                conditionals_cache,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
            )
            generated_count += 1

        if args.output.exists():
            shutil.rmtree(args.output)
        staging_output_dir.replace(args.output)
    finally:
        if staging_output_dir is not None and staging_output_dir.exists():
            shutil.rmtree(staging_output_dir)

    log_generation_summary(model_name, generated_count, len(speaker_references), str(args.output))
    log_runner_end(model_name, 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
