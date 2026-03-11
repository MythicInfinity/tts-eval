# TTS Eval

This repo runs automated TTS evals against a generated `inputs/` tree and a shared `refs/` tree.

Current runners:

- `ctc`: transcript-faithfulness via `torchaudio` `WAV2VEC2_ASR_LARGE_960H`
- `ctc_tortoise`: transcript-faithfulness via `jbetker/wav2vec2-large-robust-ft-libritts-voxpopuli` with the Tortoise tokenizer/cleaners
- `dnsmos`: no-reference quality proxy via TorchMetrics DNSMOS overall
- `nisqa`: no-reference MOS proxy via TorchMetrics NISQA MOS
- `speaker_sim`: per-utterance speaker similarity via batched SpeechBrain ECAPA embeddings
- `utmos`: per-utterance MOS prediction via the official `UTMOSv2` package
- `audiobox`: per-utterance aesthetic scores via Audiobox Aesthetics (`CE`, `PQ`)

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

Run Tortoise-backed CTC:

```bash
eval/runners/ctc_tortoise/d.sh
```

Run DNSMOS:

```bash
eval/runners/dnsmos/d.sh
```

Run NISQA:

```bash
eval/runners/nisqa/d.sh
```

Run speaker similarity:

```bash
eval/runners/speaker_sim/d.sh
```

Run UTMOS:

```bash
eval/runners/utmos/d.sh
```

Run Audiobox aesthetics:

```bash
eval/runners/audiobox/d.sh
```

Tune UTMOS batch size (default `32`) via env var:

```bash
UTMOS_BATCH_SIZE=32 eval/runners/utmos/d.sh
```

Tune UTMOS data loading workers (default `2`) via env var:

```bash
UTMOS_NUM_WORKERS=8 eval/runners/utmos/d.sh
```

Limit CPU thread fanout in numeric libs used by UTMOS preprocessing (default `1`):

```bash
UTMOS_CPU_THREADS=1 eval/runners/utmos/d.sh
```

If you see `Unexpected bus error encountered in worker` (shared memory exhaustion), increase Docker shm for UTMOS (default is `8g`):

```bash
UTMOS_SHM_SIZE=16g eval/runners/utmos/d.sh
```

To re-enable silence trimming (default is disabled), set:

```bash
UTMOS_REMOVE_SILENT_SECTION=true eval/runners/utmos/d.sh
```

Optional speed knobs for UTMOS internals (may change score characteristics):

```bash
UTMOS_SPEC_MIXUP_INNER=false UTMOS_SPEC_NUM_FRAMES=1 eval/runners/utmos/d.sh
```

Tune DNSMOS batch size (default `8`) via env var:

```bash
DNSMOS_BATCH_SIZE=16 eval/runners/dnsmos/d.sh
```

Tune NISQA batch size (default `8`) via env var:

```bash
NISQA_BATCH_SIZE=16 eval/runners/nisqa/d.sh
```

Tune speaker-sim batch size (default `8`) via env var:

```bash
SPEAKER_SIM_BATCH_SIZE=16 eval/runners/speaker_sim/d.sh
```

Tune Audiobox batch size (default `16`) via env var:

```bash
AUDIOBOX_BATCH_SIZE=32 eval/runners/audiobox/d.sh
```

Optional DNSMOS runtime tuning:

```bash
DNSMOS_DEVICE=auto DNSMOS_NUM_THREADS=8 eval/runners/dnsmos/d.sh
```

Optional speaker-sim runtime tuning:

```bash
SPEAKER_SIM_DEVICE=auto eval/runners/speaker_sim/d.sh
```

Optional NISQA runtime tuning:

```bash
NISQA_DEVICE=auto eval/runners/nisqa/d.sh
```

Optional Audiobox runtime tuning:

```bash
AUDIOBOX_DEVICE=cuda:1 eval/runners/audiobox/d.sh
```

Optional Tortoise CTC runtime tuning:

```bash
TORTOISE_CTC_DEVICE=cuda eval/runners/ctc_tortoise/d.sh
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
- `ctc_tortoise`
- `dnsmos`
- `nisqa`
- `speaker_sim`
- `utmos`
- `audiobox`

## Outputs

Runner outputs are written under the repo-level `data/evals/` tree:

- `data/evals/ctc/<model>/...`
- `data/evals/ctc_tortoise/<model>/...`
- `data/evals/dnsmos/<model>/...`
- `data/evals/nisqa/<model>/...`
- `data/evals/speaker_sim/<model>/...`
- `data/evals/utmos/<model>/...`
- `data/evals/audiobox/<model>/...`

Each runner writes timestamped JSON artifacts per model:

- `summary_<timestamp>.json`
- `metadata_<timestamp>.json`

The `ctc` runner also writes:

- `per_utt_<timestamp>.jsonl`

The `ctc_tortoise` runner also writes:

- `per_utt_<timestamp>.jsonl`

The `dnsmos` runner also writes:

- `per_utt_<timestamp>.jsonl`

The `nisqa` runner also writes:

- `per_utt_<timestamp>.jsonl`

The `speaker_sim` runner also writes:

- `per_utt_<timestamp>.jsonl`

The `utmos` runner also writes:

- `per_utt_<timestamp>.jsonl`

The `audiobox` runner also writes:

- `per_utt_<timestamp>.jsonl`

## Coalescing Results

The coalescer searches recursively, so you can point it at the repo root and it will find the current output tree.

```bash
python3 scripts/coalesce_jsons.py --eval-root . --output data/evals/coalesced_summary.json
```

The coalesced file contains one object per model and currently includes:

- `ctc_closeness_mean`
- `ctc_tortoise_closeness_mean`
- `dnsmos_ovrl_mean`
- `nisqa_mos_mean`
- `speaker_sim_ecapa_mean`
- `utmos_mean`
- `audiobox_ce_mean`
- `audiobox_pq_mean`

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

- `metric_mean` for utterance-level metrics such as `ctc`, `ctc_tortoise`, and `dnsmos`
- `metric_mean` for utterance-level metrics such as `nisqa`
- `metric_mean` for utterance-level metrics such as `speaker_sim`
- `metric_mean` for utterance-level metrics such as `utmos`
- `ce_mean` and `pq_mean` for `audiobox`
- mean values only, never median values
- the plotting command runs fully inside Docker; no host venv or system Python packages are required

## Notes

- `ctc` requires Docker with GPU access because the runner uses `--gpus all`.
- `ctc_tortoise` uses the Tortoise-author ASR checkpoint `jbetker/wav2vec2-large-robust-ft-libritts-voxpopuli`, applies the Tortoise `english_cleaners`-style transcript normalization, and records the same `exp(-ctc_loss/target_len)` closeness transform as the stock `ctc` runner.
- `dnsmos` uses the TorchMetrics functional DNSMOS API and records only the overall MOS-like output.
- `nisqa` uses the TorchMetrics functional NISQA API and records only the MOS output.
- `speaker_sim` uses `speechbrain/spkrec-ecapa-voxceleb`, averages all reference embeddings per speaker, then scores each generated utterance with cosine similarity against that speaker centroid.
- `speaker_sim` runs batched ECAPA inference (default `8`) when waveform lengths match and shows per-model tqdm progress bars.
- `utmos` uses the official `UTMOSv2` package pinned to commit `e53a6762948b908105d48d6cfd453f1b58156ed0`, runs batched `input_dir` inference with the pretrained `fusion_stage3` model, and requires Docker GPU access.
- `audiobox` uses `audiobox_aesthetics`, records `CE` and `PQ` per utterance, and requires Docker GPU access.
- `dnsmos` evaluates utterances in batches (default `8` per forward pass) when sample rate and waveform length match; it falls back to per-utterance scoring if a batch call fails.
- `nisqa` evaluates utterances in batches (default `8`) when waveform lengths match and marks the whole batch as failed on backend exceptions.
- `audiobox` evaluates utterances in batches (default `16`) and recursively splits failed batches to isolate per-utterance failures.
- `dnsmos` persists TorchMetrics model downloads by mounting the host cache path `${XDG_CACHE_HOME:-$HOME/.cache}/torchmetrics` into `/home/app/.torchmetrics`.
- `utmos` persists model downloads by mounting the host cache path `${XDG_CACHE_HOME:-$HOME/.cache}` into `/home/app/.cache` and setting `UTMOSV2_CHACHE=/home/app/.cache/utmosv2` inside the runner container.
- `audiobox` persists model downloads by mounting the host cache path `${XDG_CACHE_HOME:-$HOME/.cache}` into `/home/app/.cache` and using HF/Torch cache env vars inside the runner container.
- Invalid or unreadable WAVs are skipped during file discovery.
