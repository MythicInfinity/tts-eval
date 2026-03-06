# TTS Eval

This repo runs automated TTS evals against a generated `inputs/` tree and a shared `refs/` tree.

Current runners:

- `ctc`: transcript-faithfulness via `torchaudio` `WAV2VEC2_ASR_LARGE_960H`
- `dnsmos`: no-reference quality proxy via TorchMetrics DNSMOS overall
- `speaker_sim`: per-utterance speaker similarity via batched SpeechBrain ECAPA embeddings
- `utmos`: per-utterance MOS prediction via the official `UTMOSv2` package

## Input Layout

Generated inputs are organized by model:

```text
inputs/
  model_a/
    speaker01_00001.wav
    speaker01_00001.txt
    speaker01_00002.wav
    speaker01_00002.txt
  model_b/
    speaker01_00001.wav
    speaker01_00001.txt
```

Reference files are pooled in one directory:

```text
refs/
  speaker01_00001.wav
  speaker01_00001.txt
  speaker01_00002.wav
  speaker01_00002.txt
```

## Running One Eval

Each runner builds or reuses its own Docker image, mounts the repo root at `/app`, runs the model code as your current host UID/GID instead of root, and mounts your host cache root into the container so model downloads are reused across runs.

Run CTC:

```bash
eval/runners/ctc/d.sh
```

Run DNSMOS:

```bash
eval/runners/dnsmos/d.sh
```

Run speaker similarity:

```bash
eval/runners/speaker_sim/d.sh
```

Run UTMOS:

```bash
eval/runners/utmos/d.sh
```

Tune UTMOS batch size (default `16`) via env var:

```bash
UTMOS_BATCH_SIZE=32 eval/runners/utmos/d.sh
```

Tune DNSMOS batch size (default `8`) via env var:

```bash
DNSMOS_BATCH_SIZE=16 eval/runners/dnsmos/d.sh
```

Tune speaker-sim batch size (default `8`) via env var:

```bash
SPEAKER_SIM_BATCH_SIZE=16 eval/runners/speaker_sim/d.sh
```

Optional DNSMOS runtime tuning:

```bash
DNSMOS_DEVICE=auto DNSMOS_NUM_THREADS=8 eval/runners/dnsmos/d.sh
```

Optional speaker-sim runtime tuning:

```bash
SPEAKER_SIM_DEVICE=auto eval/runners/speaker_sim/d.sh
```

By default, all eval launchers use:

- `data/inputs`
- `data/refs`

You can still override them:

```bash
eval/runners/ctc/d.sh /abs/path/to/inputs /abs/path/to/refs
```

To force a rebuild of a runner image:

```bash
BUILD_IMAGE=1 eval/runners/ctc/d.sh
```

You can optionally pass a fixed UTC timestamp as the third argument:

```bash
eval/runners/ctc/d.sh /abs/path/to/inputs /abs/path/to/refs 2026-03-06T14:32:10Z
```

## Running The Current Eval Set

Run all implemented runners:

```bash
scripts/run_all.sh
```

This currently runs:

- `ctc`
- `dnsmos`
- `speaker_sim`
- `utmos`

## Outputs

Runner outputs are written under the repo-level `data/evals/` tree:

- `data/evals/ctc/<model>/...`
- `data/evals/dnsmos/<model>/...`
- `data/evals/speaker_sim/<model>/...`
- `data/evals/utmos/<model>/...`

Each runner writes timestamped JSON artifacts per model:

- `summary_<timestamp>.json`
- `metadata_<timestamp>.json`

The `ctc` runner also writes:

- `per_utt_<timestamp>.jsonl`

The `dnsmos` runner also writes:

- `per_utt_<timestamp>.jsonl`

The `speaker_sim` runner also writes:

- `per_utt_<timestamp>.jsonl`

The `utmos` runner also writes:

- `per_utt_<timestamp>.jsonl`

## Coalescing Results

The coalescer searches recursively, so you can point it at the repo root and it will find the current output tree.

```bash
python3 scripts/coalesce_jsons.py --eval-root . --output data/evals/coalesced_summary.json
```

The coalesced file contains one object per model and currently includes:

- `ctc_closeness_mean`
- `dnsmos_ovrl_mean`
- `speaker_sim_ecapa_mean`
- `utmos_mean`

## Plotting Mean Results

Generate a grouped bar chart of the latest mean evals, grouped by eval metric with one bar per model:

```bash
scripts/plot_eval_means/d.sh --eval-root . --output data/evals/mean_eval_plot.png
```

Use the previous layout with evals grouped under each model:

```bash
scripts/plot_eval_means/d.sh --eval-root . --output data/evals/mean_eval_plot.png --group-by-model
```

Add stddev error bars where the metric exposes `metric_std`:

```bash
scripts/plot_eval_means/d.sh --eval-root . --output data/evals/mean_eval_plot.png --include-stddev
```

The plot uses:

- `metric_mean` for utterance-level metrics such as `ctc` and `dnsmos`
- `metric_mean` for utterance-level metrics such as `speaker_sim`
- `metric_mean` for utterance-level metrics such as `utmos`
- mean values only, never median values
- the plotting command runs fully inside Docker; no host venv or system Python packages are required

## Notes

- `ctc` requires Docker with GPU access because the runner uses `--gpus all`.
- `dnsmos` uses the TorchMetrics functional DNSMOS API and records only the overall MOS-like output.
- `speaker_sim` uses `speechbrain/spkrec-ecapa-voxceleb`, averages all reference embeddings per speaker, then scores each generated utterance with cosine similarity against that speaker centroid.
- `speaker_sim` runs batched ECAPA inference (default `8`) when waveform lengths match and shows per-model tqdm progress bars.
- `utmos` uses the official `UTMOSv2` package pinned to commit `e53a6762948b908105d48d6cfd453f1b58156ed0`, runs batched `input_dir` inference with the pretrained `fusion_stage3` model, and requires Docker GPU access.
- `dnsmos` evaluates utterances in batches (default `8` per forward pass) when sample rate and waveform length match; it falls back to per-utterance scoring if a batch call fails.
- `dnsmos` persists TorchMetrics model downloads by mounting the host cache path `${XDG_CACHE_HOME:-$HOME/.cache}/torchmetrics` into `/home/app/.torchmetrics`.
- `utmos` persists model downloads by mounting the host cache path `${XDG_CACHE_HOME:-$HOME/.cache}` into `/home/app/.cache` and setting `UTMOSV2_CHACHE=/home/app/.cache/utmosv2` inside the runner container.
- Invalid or unreadable WAVs are skipped during file discovery.
