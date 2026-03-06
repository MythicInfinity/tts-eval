Below is the complete updated implementation spec with all requested changes incorporated.

---

# Automated TTS Evaluation System — Complete Implementation Spec

## 1. Goal

Build a **containerized, repeatable evaluation pipeline** that scores multiple TTS systems from generated WAV directories and produces:

* **per-metric per-model JSON summaries**
* **one coalesced summary JSON** with **one object per model**
* enough metadata to support **latest-run-wins** behavior and reproducibility

The evaluation uses a small set of **perception-adjacent automated metrics**, plus one text-faithfulness metric and one speaker-similarity metric.

## 2. Final metric set

The implemented metric set is:

1. **CTC closeness**
2. **TTSDS2 total**
3. **DNSMOS overall**
4. **Speaker similarity**
5. **UTMOS**

We are **not** including PESQ, hand-built prosody metrics, Audiobox Aesthetics, or SQUIM in v1.

## 3. Scope and assumptions

### 3.1 Language scope

This system is **English-only in v1**.

### 3.2 ASR backend scope

The CTC backend is fixed to:

* **torchaudio `WAV2VEC2_ASR_LARGE_960H`**

### 3.3 Core design intent

The pipeline is intended to compare multiple generated TTS model outputs in a uniform way by:

* walking model directories
* evaluating all valid utterances found within them
* producing reproducible machine-readable summaries
* supporting repeated runs where **latest timestamped output wins**

## 4. Final rationale for included metrics

### 4.1 CTC closeness

Included as a transcript-faithfulness signal.

### 4.2 TTSDS2

Included as a **distributional**, model-level perceptual benchmark score.

### 4.3 DNSMOS

Included as a no-reference signal-quality / artifact proxy.

### 4.4 Speaker similarity

Included as an explicit per-utterance speaker-matching score using a standard ECAPA embedding model.

### 4.5 UTMOS

Included as a dedicated MOS-prediction signal for speech naturalness / perceived quality.

## 5. Metrics and exact backend choices

## 5.1 CTC closeness

**Purpose**
Measure how well synthesized audio matches the intended transcript, using a smooth scalar rather than only decoded WER.

**Chosen backend**

* **torchaudio `WAV2VEC2_ASR_LARGE_960H`**

**Metric name**

* `ctc_closeness`

**Computation**

Given:

* generated waveform `x`
* transcript text from a filename-matched sibling `.txt`
* tokenized target sequence `y`

Compute a normalized CTC score such as:

`ctc_closeness = exp(-ctc_loss(x, y) / len(y))`

Implementation rules:

* the exact transform must be fixed and versioned
* no custom text normalization is required
* tokenize transcript text according to the CTC tokenizer path
* discard unknown characters during tokenization
* higher is better

**Required inputs**

* generated WAV
* sibling transcript `.txt`

**Optional debug fields**

* decoded transcript
* normalized CTC loss
* WER/CER for debugging only

---

## 5.2 TTSDS2

**Purpose**
Provide a broad synthetic-speech quality score incorporating perceptual dimensions beyond raw ASR faithfulness.

**Chosen backend**

* current official `ttsds` package / TTSDS2 family

**Metric name**

* `ttsds2_total`

**Critical semantic clarification**

TTSDS2 is **not** treated like the other utterance-level metrics. It is a **distributional score over sets of audio**, so the runner operates at the **model level**, not the utterance level.

**How to handle multiple speakers**

For v1, each model directory is treated as a **single pooled generated speech distribution** containing all speakers present in that model.

The reference side is the pooled `/refs` speech distribution.

The TTSDS2 runner computes:

* one aggregate score for each model’s generated set
* always against the same shared `/refs` set

This means TTSDS2 evaluates whether the **overall generated distribution** resembles the reference speech distribution. It does **not** try to perform utterance-wise or speaker-wise one-to-one matching.

**Speaker-category handling**

TTSDS2 includes a speaker-related component, but this pipeline already has a dedicated speaker metric:

* `speaker_sim_ecapa`

To avoid double-counting speaker similarity, the TTSDS2 runner must use a **custom category weighting** with the speaker term disabled.

Required weights for v1:

```python
{
    "SPEAKER": 0.0,
    "INTELLIGIBILITY": 1/3,
    "PROSODY": 1/3,
    "GENERIC": 1/3,
    "ENVIRONMENT": 0.0,
}
```

Operational meaning:

* `SPEAKER = 0.0` because speaker identity is evaluated separately by ECAPA
* `INTELLIGIBILITY / PROSODY / GENERIC` are weighted evenly
* `ENVIRONMENT = 0.0` to keep v1 focused on core TTS behavior

Therefore, `ttsds2_total` in this system means:

> a model-level distributional TTSDS2 aggregate with speaker and environment disabled, and equal weighting over intelligibility, prosody, and generic quality

**Required inputs**

* generated WAV set for one model
* shared reference WAV set from `/refs`

**Per-utterance output**

* not required
* TTSDS2 is model-level in this pipeline

---

## 5.3 DNSMOS

**Purpose**
Provide a widely used no-reference quality proxy, useful for catching degradations, artifacts, or poor signal quality.

**Chosen backend**

* `torchmetrics.audio.dnsmos.DeepNoiseSuppressionMeanOpinionScore`

**Metric name**

* `dnsmos_ovrl`

**Computation**

* use the overall MOS-like output
* ignore SIG/BAK in v1 unless needed for debugging
* compute per-sample values using the appropriate per-sample path rather than only a batch-reduced class output

**Required inputs**

* generated WAV only

---

## 5.4 Speaker similarity

**Purpose**
Measure whether generated speech sounds like the intended speaker.

**Chosen backend**

* SpeechBrain ECAPA-TDNN
* checkpoint: `speechbrain/spkrec-ecapa-voxceleb`

**Metric name**

* `speaker_sim_ecapa`

**Computation**

For each utterance:

1. parse `speaker_id` from the filename stem
2. gather all matching reference WAVs in `/refs` for that speaker
3. extract embedding for generated utterance
4. extract embeddings for all gathered reference WAVs
5. average the reference embeddings
6. compute cosine similarity between generated embedding and averaged reference embedding

If no matching reference WAVs exist for that speaker, skip the utterance for this metric.

**Required inputs**

* generated WAV
* one or more matching reference speaker WAVs from `/refs`

---

## 5.5 UTMOS

**Purpose**
Provide a dedicated MOS-prediction signal tailored to speech naturalness / perceived quality.

**Chosen backend**

* UTMOSv2

**Metric name**

* `utmos`

**Computation**

* use the package’s default prediction path on each WAV
* export one scalar per utterance

**Required inputs**

* generated WAV only

## 6. Metrics not in scope for v1

The following are explicitly out of scope:

* PESQ
* hand-built prosody metrics
* Audiobox Aesthetics
* SQUIM / SQUIM_SUBJECTIVE
* human listening tests
* manual MOS collection
* pairwise significance testing
* composite weighted leaderboard scores
* prosody-specific dashboards

## 7. Input assumptions

## 7.1 Mounted paths

The pipeline mounts:

* generated base directory to **`/inputs`**
* reference base directory to **`/refs`**
* output base directory to **`/output`**

## 7.2 Generated directory layout

The pipeline assumes `/inputs` contains **immediate subdirectories**, each representing one TTS system / model.

Example:

```text
/inputs/
  model_a/
    speaker01_00001.wav
    speaker01_00001.txt
    speaker01_00002.wav
    speaker01_00002.txt
    speaker02_00001.wav
    speaker02_00001.txt
  model_b/
    speaker01_00001.wav
    speaker01_00001.txt
```

Each utterance is represented by a filename-matched pair:

* `speakerid_{idx:05d}.wav`
* `speakerid_{idx:05d}.txt`

Example:

* `speaker01_00001.wav`
* `speaker01_00001.txt`

## 7.3 Reference directory layout

The reference directory `/refs` uses the same convention as utterances.

Example:

```text
/refs/
  speaker01_00001.wav
  speaker01_00001.txt
  speaker01_00002.wav
  speaker01_00002.txt
  speaker02_00001.wav
  speaker02_00001.txt
```

No CSV manifests are required for:

* transcripts
* speaker tracking
* utterance-to-speaker mapping
* reference lookup

## 8. Filename conventions and matching rules

## 8.1 Utterance id

The utterance id is the filename stem.

Example:

* WAV: `speaker01_00017.wav`
* transcript: `speaker01_00017.txt`

## 8.2 Speaker id

The speaker id is extracted from the stem prefix before the final underscore-delimited numeric index.

Example:

* stem: `speaker01_00017`
* speaker id: `speaker01`

No separate speaker manifest is used.

## 8.3 Transcript lookup

For a generated WAV:

* transcript path is the same path with `.txt` extension

Example:

* `/inputs/model_a/speaker01_00017.wav`
* `/inputs/model_a/speaker01_00017.txt`

If the `.txt` file is missing, transcript-dependent metrics skip that utterance.

## 8.4 Reference lookup

For speaker similarity, the runner selects all reference WAVs in `/refs` whose stems begin with the speaker id followed by `_`.

Example for `speaker01`:

* `/refs/speaker01_00001.wav`
* `/refs/speaker01_00002.wav`

Matching reference `.txt` files may exist for completeness, but speaker similarity does not require them.

## 9. Reference audio requirements

For speaker similarity:

* at least one reference WAV per speaker is required
* multiple reference WAVs per speaker are supported and preferred

Recommended targets:

* **minimum useful**: ~10–15 seconds total speech per speaker
* **preferred**: ~20–30 seconds total speech per speaker
* **better later**: 30–60 seconds across several clips

For TTSDS2:

* `/refs` acts as the shared real/reference speech distribution for all models
* the same `/refs` set must be used for every model in a comparable evaluation batch

## 10. Data volume expectations

Per model, around **1 hour of generated audio** is enough for meaningful system-level averages, assuming clips are not extremely repetitive.

The pipeline should report:

* `n_utts`
* `total_audio_sec`

For TTSDS2, these fields refer to the **generated set used for that model-level distributional run**.

## 11. Output contract

The system uses **JSON over CSV**.

## 11.1 Timestamping

Every output summary must include:

* a timestamp in the **filename**
* the same timestamp inside the JSON payload as `run_timestamp_utc`

Recommended filename pattern:

* `summary_2026-03-06T14-32-10Z.json`

Filename timestamp is sufficient for **latest-run-wins** behavior.

## 11.2 Per-metric per-model summary JSON

Each metric runner writes one summary JSON per model.

Example path:

```text
/output/dnsmos/model_a/summary_2026-03-06T14-32-10Z.json
```

Recommended schema:

```json
{
  "run_timestamp_utc": "2026-03-06T14:32:10Z",
  "metric_name": "dnsmos_ovrl",
  "metric_version": "torchmetrics_dnsmos_<version>",
  "model": "model_a",
  "n_utts": 1240,
  "total_audio_sec": 3611.2,
  "metric_mean": 3.41,
  "metric_median": 3.44,
  "metric_std": 0.21,
  "fail_count": 0,
  "skip_count": 0
}
```

For reproducibility, `metric_version` should capture:

* package version
* checkpoint / bundle name if relevant
* any non-default weighting or configuration that changes semantics

Examples:

* `torchaudio_WAV2VEC2_ASR_LARGE_960H`
* `speechbrain/spkrec-ecapa-voxceleb`
* `torchmetrics_dnsmos_<version>`
* `sarulab-speech/UTMOSv2@<commit>`
* `ttsds_<version_or_commit>|weights:speaker=0.0,intelligibility=0.333333,prosody=0.333333,generic=0.333333,environment=0.0`

## 11.3 Optional per-utterance JSONL

Each utterance-level runner may also write a per-utterance JSONL file for debugging.

Recommended path:

```text
/output/ctc/model_a/per_utt_2026-03-06T14-32-10Z.jsonl
```

Recommended record schema:

```json
{"run_timestamp_utc":"2026-03-06T14:32:10Z","metric_name":"ctc_closeness","metric_version":"torchaudio_WAV2VEC2_ASR_LARGE_960H","model":"model_a","utt_id":"speaker01_00001","wav_path":"/inputs/model_a/speaker01_00001.wav","metric_value":0.923,"status":"ok","error":null}
```

TTSDS2 does **not** require per-utterance JSONL because it is distributional and model-level.

## 11.4 Final coalesced summary JSON

The outer coalescer produces one final JSON file containing one object per model.

Recommended shape:

```json
[
  {
    "run_timestamp_utc": "2026-03-06T14:32:10Z",
    "model": "model_a",
    "n_utts": 1240,
    "total_audio_sec": 3611.2,
    "ctc_closeness_mean": 0.91,
    "ttsds2_total": 0.78,
    "dnsmos_ovrl_mean": 3.41,
    "speaker_sim_ecapa_mean": 0.72,
    "utmos_mean": 3.89
  }
]
```

## 12. System architecture

## 12.1 Directory structure

```text
eval/
  runners/
    ctc/
      Dockerfile
      d.sh
      run_inner.py
    ttsds2/
      Dockerfile
      d.sh
      run_inner.py
    dnsmos/
      Dockerfile
      d.sh
      run_inner.py
    speaker_sim/
      Dockerfile
      d.sh
      run_inner.py
    utmos/
      Dockerfile
      d.sh
      run_inner.py
  scripts/
    run_all.sh
    coalesce_jsons.py
```

## 13. Container contract

Each metric runner has the same outer contract.

## 13.1 `d.sh`

Responsible for:

* building or using the Docker image
* mounting generated base dir to `/inputs`
* mounting reference base dir to `/refs`
* mounting output dir to `/output`
* invoking `run_inner.py`

## 13.2 `run_inner.py`

Responsible for:

* iterating over immediate subdirectories in `/inputs`
* treating each subdirectory as one model
* evaluating WAVs inside that model directory
* using sibling `.txt` files when needed
* using `/refs` for speaker/reference resources
* writing per-model JSON outputs
* optionally writing per-utterance JSONL outputs

This separation keeps outer orchestration simple and metric-specific dependencies isolated.

## 14. Orchestration

## 14.1 `run_all.sh`

Responsibilities:

* invoke every metric runner
* pass the same `/inputs`, `/refs`, and `/output` mounts
* ensure each runner writes to its own output subtree

Example output tree:

```text
/output/
  ctc/
    model_a/
      summary_<timestamp>.json
  ttsds2/
    model_a/
      summary_<timestamp>.json
  dnsmos/
    model_a/
      summary_<timestamp>.json
  speaker_sim/
    model_a/
      summary_<timestamp>.json
  utmos/
    model_a/
      summary_<timestamp>.json
```

## 14.2 `coalesce_jsons.py`

Responsibilities:

* walk all metric output trees
* choose the **latest timestamped summary filename** per model per metric
* merge the latest summaries across metrics
* output one final summary JSON with one object per model

Filename timestamp is the source of truth for latest-run-wins selection.

## 15. Scoring logic by runner

## 15.1 CTC runner

For each utterance:

* load WAV
* load sibling transcript `.txt`
* tokenize transcript using the CTC tokenizer path
* discard unknown characters
* resample if needed
* compute normalized CTC loss / transformed closeness score
* append per-utterance result

Aggregate by model:

* mean
* median
* std
* fail count
* skip count

## 15.2 TTSDS2 runner

For each model:

* gather all valid generated WAVs in `/inputs/<model>/`
* gather the shared reference WAV set in `/refs/`
* run TTSDS2 on the **generated distribution vs reference distribution**
* apply the fixed category weights:

```python
{
    "SPEAKER": 0.0,
    "INTELLIGIBILITY": 1/3,
    "PROSODY": 1/3,
    "GENERIC": 1/3,
    "ENVIRONMENT": 0.0,
}
```

* write one aggregate summary JSON for the model

Aggregate by model:

* one distributional score
* optional auxiliary outputs if the package exposes them
* fail count / skip count over file discovery and loading

## 15.3 DNSMOS runner

For each utterance:

* load WAV
* resample to required rate if needed
* compute per-sample DNSMOS overall
* append result

Aggregate by model:

* mean
* median
* std
* fail count
* skip count

## 15.4 Speaker-sim runner

For each utterance:

* load WAV
* parse speaker id from filename
* gather `/refs/<speaker_id>_*.wav`
* compute ECAPA embeddings
* compute cosine similarity
* append result

Aggregate by model:

* mean
* median
* std
* fail count
* skip count

## 15.5 UTMOS runner

For each utterance:

* load WAV
* resample if required
* run UTMOSv2
* append result

Aggregate by model:

* mean
* median
* std
* fail count
* skip count

## 16. File matching and skip behavior

WAV matching is based on **filename stem**.

The system must:

* skip files with missing required sidecar `.txt` for transcript-dependent metrics
* skip files with missing speaker refs for speaker similarity
* skip unreadable or empty WAVs
* count skipped items in `skip_count`
* count backend/runtime failures in `fail_count`
* log errors clearly

`skip_count` and `fail_count` are distinct and must not be collapsed together.

## 17. Normalization and audio handling

All runners share the same basic audio handling rules:

* accept mono WAV as canonical input
* downmix multi-channel audio to mono
* resample as needed for the target backend
* reject unreadable or empty files
* preserve natural silence unless a metric explicitly requires otherwise

The exact sample-rate target may differ by backend, so resampling happens inside each runner.

## 18. Versioning and reproducibility

Every summary must carry:

* `run_timestamp_utc`
* `metric_name`
* `metric_version`

Every runner should also emit a lightweight metadata JSON containing:

* image tag
* Python version
* package versions
* model bundle / checkpoint identifier
* any non-default config affecting semantics

This is especially important because:

* torchaudio bundles are versioned
* TorchMetrics behavior can change across versions
* UTMOSv2 and TTSDS2 are GitHub-hosted projects under active development
* TTSDS2 custom weighting changes metric meaning and must be recorded

## 19. Recommended concrete model identifiers

### 19.1 CTC closeness

* **Backend**: `torchaudio.pipelines.WAV2VEC2_ASR_LARGE_960H`

### 19.2 TTSDS2

* **Backend**: official `ttsds` package / current TTSDS2 line
* **Weighting**:

  * `SPEAKER: 0.0`
  * `INTELLIGIBILITY: 1/3`
  * `PROSODY: 1/3`
  * `GENERIC: 1/3`
  * `ENVIRONMENT: 0.0`

### 19.3 DNSMOS

* **Backend**: `torchmetrics.audio.dnsmos.DeepNoiseSuppressionMeanOpinionScore`
* **Per-sample path**: use the per-sample interface rather than relying only on batch-reduced output

### 19.4 Speaker similarity

* **Backend**: `speechbrain/spkrec-ecapa-voxceleb`

### 19.5 UTMOS

* **Backend**: `sarulab-speech/UTMOSv2`

## 20. Non-goals for v1

The following are explicitly outside the first implementation:

* human listening tests
* manual MOS collection
* pairwise significance testing
* prosody-specific subscore dashboards
* Audiobox Aesthetics integration
* SQUIM integration
* composite weighted leaderboard score

A composite score can be added later, but v1 exposes raw metric outputs only.

## 21. Final deliverable definition

The implementation is complete when the following exists:

* five metric runner subdirectories:

  * `ctc`
  * `ttsds2`
  * `dnsmos`
  * `speaker_sim`
  * `utmos`
* each runner has:

  * `Dockerfile`
  * `d.sh`
  * `run_inner.py`
* one outer orchestrator:

  * `scripts/run_all.sh`
* one coalescer:

  * `scripts/coalesce_jsons.py`
* one final JSON artifact:

  * one object per model
  * latest run selected per model per metric
  * fields for the five scores plus metadata

## 22. Final recommended summary schema

Use this as the canonical coalesced output object shape:

```json
{
  "run_timestamp_utc": "2026-03-06T14:32:10Z",
  "model": "model_a",
  "n_utts": 1240,
  "total_audio_sec": 3611.2,
  "ctc_closeness_mean": 0.91,
  "ttsds2_total": 0.78,
  "dnsmos_ovrl_mean": 3.41,
  "speaker_sim_ecapa_mean": 0.72,
  "utmos_mean": 3.89
}
```

## 23. Important semantic note for downstream consumers

The final summary contains both:

* metrics that are **aggregates over utterance-level scores**
* one metric that is a **model-level distributional score**

Specifically:

* `ctc_closeness_mean` = mean over utterances
* `dnsmos_ovrl_mean` = mean over utterances
* `speaker_sim_ecapa_mean` = mean over utterances
* `utmos_mean` = mean over utterances
* `ttsds2_total` = **single model-level distributional score**, not an utterance mean

That difference is intentional and should be preserved in implementation notes and downstream reporting.
