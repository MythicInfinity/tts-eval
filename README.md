# TTS Eval

This repo runs automated TTS evals against a generated `inputs/` tree and a shared `refs/` tree.

Current runners:

- `ctc`: transcript-faithfulness via `torchaudio` `WAV2VEC2_ASR_LARGE_960H`
- `ttsds2`: model-level distributional score via the official `ttsds` package
- `dnsmos`: no-reference quality proxy via TorchMetrics DNSMOS overall

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

Each runner builds or reuses its own Docker image and writes outputs under its own directory because `/app` is the runner directory.

Run CTC:

```bash
eval/runners/ctc/d.sh /abs/path/to/inputs /abs/path/to/refs
```

Run TTSDS2:

```bash
eval/runners/ttsds2/d.sh /abs/path/to/inputs /abs/path/to/refs
```

Run DNSMOS:

```bash
eval/runners/dnsmos/d.sh /abs/path/to/inputs /abs/path/to/refs
```

To force a rebuild of a runner image:

```bash
BUILD_IMAGE=1 eval/runners/ctc/d.sh /abs/path/to/inputs /abs/path/to/refs
```

You can optionally pass a fixed UTC timestamp as the third argument:

```bash
eval/runners/ctc/d.sh /abs/path/to/inputs /abs/path/to/refs 2026-03-06T14:32:10Z
```

## Running The Current Eval Set

Run all implemented runners:

```bash
scripts/run_all.sh /abs/path/to/inputs /abs/path/to/refs
```

This currently runs:

- `ctc`
- `ttsds2`
- `dnsmos`

## Outputs

Runner outputs are written under each runner directory:

- `eval/runners/ctc/data/evals/ctc/<model>/...`
- `eval/runners/ttsds2/data/evals/ttsds2/<model>/...`
- `eval/runners/dnsmos/data/evals/dnsmos/<model>/...`

Each runner writes timestamped JSON artifacts per model:

- `summary_<timestamp>.json`
- `metadata_<timestamp>.json`

The `ctc` runner also writes:

- `per_utt_<timestamp>.jsonl`

The `dnsmos` runner also writes:

- `per_utt_<timestamp>.jsonl`

## Coalescing Results

The coalescer searches recursively, so you can point it at the repo root and it will find runner-local `data/evals/...` directories.

```bash
python3 scripts/coalesce_jsons.py --eval-root . --output data/evals/coalesced_summary.json
```

The coalesced file contains one object per model and currently includes:

- `ctc_closeness_mean`
- `ttsds2_total`
- `dnsmos_ovrl_mean`

## Notes

- `ctc` requires Docker with GPU access because the runner uses `--gpus all`.
- `ttsds2` uses the official `ttsds` package with fixed category weights:
  - `SPEAKER=0.0`
  - `INTELLIGIBILITY=1/3`
  - `PROSODY=1/3`
  - `GENERIC=1/3`
  - `ENVIRONMENT=0.0`
- `dnsmos` uses the TorchMetrics functional DNSMOS API and records only the overall MOS-like output.
- Invalid or unreadable WAVs are skipped during file discovery.
