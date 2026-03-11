from __future__ import annotations

import argparse
import sys
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

from tts_eval.coalesce import collect_latest_summaries
from tts_eval.plotting import render_mean_plot_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot grouped mean eval bars from the latest summary JSONs."
    )
    parser.add_argument("--eval-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("data/evals/mean_eval_plot.png"))
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--include-stddev", action="store_true")
    parser.add_argument(
        "--group-by-model",
        action="store_true",
        help="Deprecated no-op retained for backward compatibility.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    latest_summaries = collect_latest_summaries(args.eval_root)
    try:
        output_paths = render_mean_plot_outputs(
            latest_summaries=latest_summaries,
            output_path=args.output,
            title=args.title,
            include_stddev=args.include_stddev,
            dpi=args.dpi,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    except RuntimeError as exc:
        raise SystemExit(f"{exc}. Plot image requires matplotlib and numpy in the runtime.") from exc

    print(f"saved combined plot PNG to {output_paths.combined_png}", flush=True)
    print(f"saved combined plot SVG to {output_paths.combined_svg}", flush=True)
    for metric_label in output_paths.metric_pngs:
        print(f"saved {metric_label} PNG to {output_paths.metric_pngs[metric_label]}", flush=True)
        print(f"saved {metric_label} SVG to {output_paths.metric_svgs[metric_label]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
