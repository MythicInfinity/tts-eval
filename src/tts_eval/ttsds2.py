from __future__ import annotations

import contextlib
import inspect
import os
import platform
import shutil
import tempfile
import wave
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from tts_eval.discovery import iter_wavs


TTSDS2_WEIGHTS = {
    "SPEAKER": 0.0,
    "INTELLIGIBILITY": 1.0 / 3.0,
    "PROSODY": 1.0 / 3.0,
    "GENERIC": 1.0 / 3.0,
    "ENVIRONMENT": 0.0,
}


@dataclass(frozen=True)
class WavInfo:
    path: Path
    duration_sec: float


@dataclass(frozen=True)
class TTSDS2Runtime:
    BenchmarkSuite: Any
    DirectoryDataset: Any
    BenchmarkCategory: Any
    package_version: str
    metric_version: str


@dataclass(frozen=True)
class TTSDS2Result:
    metric_value: float | None
    category_scores: dict[str, float]
    raw_result: dict[str, Any]


def _ttsds_cache_dir() -> Path:
    return Path(os.getenv("TTSDS_CACHE_DIR", Path.home() / ".cache" / "ttsds"))


def _is_git_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(128).startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def _repair_ttsds_noise_reference_cache() -> None:
    noise_reference_dir = _ttsds_cache_dir() / "noise-reference"
    if not noise_reference_dir.exists():
        return

    tarballs = list(noise_reference_dir.glob("*.tar.gz"))
    if any(_is_git_lfs_pointer(tarball) for tarball in tarballs):
        shutil.rmtree(noise_reference_dir, ignore_errors=True)


def _ttsds_package_version(module: Any) -> str:
    package_version = getattr(module, "__version__", None)
    if isinstance(package_version, str) and package_version:
        return package_version

    try:
        return metadata.version("ttsds")
    except metadata.PackageNotFoundError:
        return "unknown"


def _patch_hf_hub_download_auth_token_compat(huggingface_hub: Any) -> None:
    hf_hub_download = getattr(huggingface_hub, "hf_hub_download", None)
    if hf_hub_download is None:
        return

    try:
        parameters = inspect.signature(hf_hub_download).parameters
    except (TypeError, ValueError):
        return

    if "use_auth_token" in parameters:
        return

    def hf_hub_download_compat(*args: Any, **kwargs: Any) -> Any:
        use_auth_token = kwargs.pop("use_auth_token", None)
        if use_auth_token is not None and "token" not in kwargs:
            kwargs["token"] = use_auth_token
        return hf_hub_download(*args, **kwargs)

    huggingface_hub.hf_hub_download = hf_hub_download_compat


def load_ttsds2_runtime() -> TTSDS2Runtime:
    _repair_ttsds_noise_reference_cache()
    # Torch 2.6 defaults torch.load(..., weights_only=True), but TTSDS2 dependencies
    # still load trusted full checkpoints without explicitly setting weights_only.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    try:
        import huggingface_hub
        _patch_hf_hub_download_auth_token_compat(huggingface_hub)
        import ttsds
        from ttsds import BenchmarkSuite
        from ttsds.benchmarks.benchmark import BenchmarkCategory
        from ttsds.util.dataset import DirectoryDataset
    except ModuleNotFoundError as exc:
        raise RuntimeError("ttsds and huggingface_hub must be installed in the runner environment") from exc

    package_version = _ttsds_package_version(ttsds)
    metric_version = (
        f"ttsds_{package_version}"
        f"|weights:speaker={TTSDS2_WEIGHTS['SPEAKER']:.6f}"
        f",intelligibility={TTSDS2_WEIGHTS['INTELLIGIBILITY']:.6f}"
        f",prosody={TTSDS2_WEIGHTS['PROSODY']:.6f}"
        f",generic={TTSDS2_WEIGHTS['GENERIC']:.6f}"
        f",environment={TTSDS2_WEIGHTS['ENVIRONMENT']:.6f}"
        f"|include_environment=false"
    )
    return TTSDS2Runtime(
        BenchmarkSuite=BenchmarkSuite,
        DirectoryDataset=DirectoryDataset,
        BenchmarkCategory=BenchmarkCategory,
        package_version=package_version,
        metric_version=metric_version,
    )


def inspect_wav(path: Path) -> WavInfo:
    try:
        with contextlib.closing(wave.open(str(path), "rb")) as handle:
            frame_count = handle.getnframes()
            sample_rate = handle.getframerate()
    except (wave.Error, EOFError, OSError) as exc:
        raise ValueError(f"unreadable wav: {exc}") from exc

    if frame_count <= 0 or sample_rate <= 0:
        raise ValueError("empty wav")

    return WavInfo(path=path, duration_sec=frame_count / sample_rate)


def collect_valid_wavs(directory: Path) -> tuple[list[WavInfo], int]:
    valid: list[WavInfo] = []
    skip_count = 0
    for wav_path in iter_wavs(directory):
        try:
            valid.append(inspect_wav(wav_path))
        except ValueError:
            skip_count += 1
    return valid, skip_count


def stage_wavs(dataset: list[WavInfo], staging_dir: Path) -> Path:
    staging_dir.mkdir(parents=True, exist_ok=True)
    for wav_info in dataset:
        target = staging_dir / wav_info.path.name
        try:
            os.symlink(wav_info.path, target)
        except OSError:
            shutil.copy2(wav_info.path, target)
    return staging_dir


def _build_category_weights(runtime: TTSDS2Runtime) -> dict[Any, float]:
    category_enum = runtime.BenchmarkCategory
    return {
        category_enum.SPEAKER: TTSDS2_WEIGHTS["SPEAKER"],
        category_enum.INTELLIGIBILITY: TTSDS2_WEIGHTS["INTELLIGIBILITY"],
        category_enum.PROSODY: TTSDS2_WEIGHTS["PROSODY"],
        category_enum.GENERIC: TTSDS2_WEIGHTS["GENERIC"],
        category_enum.ENVIRONMENT: TTSDS2_WEIGHTS["ENVIRONMENT"],
    }


def _normalize_category_scores(raw_result: dict[str, Any]) -> dict[str, float]:
    category_scores: dict[str, float] = {}
    for key, value in raw_result.items():
        if isinstance(value, (int, float)) and key.lower() not in {"total", "weighted_total", "score"}:
            category_scores[key.lower()] = float(value)
    return category_scores


def _extract_total_result(raw_result: Any) -> dict[str, Any]:
    if isinstance(raw_result, dict):
        if any(isinstance(raw_result.get(key), (int, float)) for key in ("total", "weighted_total", "score")):
            return raw_result
        if len(raw_result) == 1:
            value = next(iter(raw_result.values()))
            if isinstance(value, dict):
                return value
    raise RuntimeError("ttsds aggregated result did not expose a per-dataset total score")


def run_ttsds2_benchmark(generated_dir: Path, refs_dir: Path, runtime: TTSDS2Runtime) -> TTSDS2Result:
    generated_dataset = runtime.DirectoryDataset(str(generated_dir), name=generated_dir.name)
    reference_dataset = runtime.DirectoryDataset(str(refs_dir), name="reference")
    suite = runtime.BenchmarkSuite(
        datasets=[generated_dataset],
        reference_datasets=[reference_dataset],
        category_weights=_build_category_weights(runtime),
        include_environment=False,
        skip_errors=True,
    )
    suite.run()
    raw_result = suite.get_aggregated_results()
    total_result = _extract_total_result(raw_result)

    metric_value = None
    if "total" in total_result and isinstance(total_result["total"], (int, float)):
        metric_value = float(total_result["total"])
    elif "weighted_total" in total_result and isinstance(total_result["weighted_total"], (int, float)):
        metric_value = float(total_result["weighted_total"])
    elif "score" in total_result and isinstance(total_result["score"], (int, float)):
        metric_value = float(total_result["score"])

    if metric_value is None:
        raise RuntimeError("ttsds result did not expose a total score")

    return TTSDS2Result(
        metric_value=metric_value,
        category_scores=_normalize_category_scores(total_result),
        raw_result=raw_result if isinstance(raw_result, dict) else {"value": raw_result},
    )


def evaluate_model(
    model: str,
    model_dir: Path,
    refs_dir: Path,
    runtime: TTSDS2Runtime,
    run_timestamp_utc: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    generated_wavs, generated_skip_count = collect_valid_wavs(model_dir)
    reference_wavs, reference_skip_count = collect_valid_wavs(refs_dir)
    skip_count = generated_skip_count + reference_skip_count
    total_audio_sec = round(sum(wav.duration_sec for wav in generated_wavs), 6)
    n_utts = len(generated_wavs)

    summary_payload = {
        "run_timestamp_utc": run_timestamp_utc,
        "metric_name": "ttsds2_total",
        "metric_version": runtime.metric_version,
        "model": model,
        "n_utts": n_utts,
        "total_audio_sec": total_audio_sec,
        "metric_value": None,
        "fail_count": 0,
        "skip_count": skip_count,
        "category_scores": {},
        "error": None,
    }
    metadata_payload = {
        "run_timestamp_utc": run_timestamp_utc,
        "metric_name": "ttsds2_total",
        "metric_version": runtime.metric_version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "ttsds_version": runtime.package_version,
        "weights": TTSDS2_WEIGHTS,
        "include_environment": False,
        "skip_errors": True,
        "generated_valid_wavs": len(generated_wavs),
        "reference_valid_wavs": len(reference_wavs),
        "generated_skip_count": generated_skip_count,
        "reference_skip_count": reference_skip_count,
    }

    if not generated_wavs:
        summary_payload["error"] = "no valid generated wavs"
        return summary_payload, metadata_payload

    if not reference_wavs:
        summary_payload["error"] = "no valid reference wavs"
        return summary_payload, metadata_payload

    with tempfile.TemporaryDirectory(prefix=f"ttsds2_{model}_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        staged_generated = stage_wavs(generated_wavs, tmpdir_path / "generated")
        staged_refs = stage_wavs(reference_wavs, tmpdir_path / "refs")
        try:
            result = run_ttsds2_benchmark(staged_generated, staged_refs, runtime)
        except Exception as exc:
            summary_payload["fail_count"] = n_utts
            summary_payload["error"] = str(exc)
            return summary_payload, metadata_payload

    summary_payload["metric_value"] = result.metric_value
    summary_payload["category_scores"] = result.category_scores
    metadata_payload["raw_result"] = result.raw_result
    return summary_payload, metadata_payload
