from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


@dataclass(frozen=True)
class HFStreamingTextDatasetSpec:
    path: str
    split: str = "train"
    name: str | None = None
    text_field: str = "text"
    transform: Callable[[Mapping[str, Any]], str] | None = None
    load_kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StreamingTextDataset(Iterable[str]):
    dataset: Any

    def __iter__(self) -> Iterator[str]:
        for example in self.dataset:
            if not isinstance(example, dict):
                raise TypeError("streaming text dataset must yield dictionaries before string projection")
            text = example.get("text")
            if not isinstance(text, str):
                raise TypeError("streaming text dataset examples must expose a string 'text' field")
            yield text


def _build_text_mapper(spec: HFStreamingTextDatasetSpec) -> Callable[[Mapping[str, Any]], dict[str, str]]:
    def to_text(example: Mapping[str, Any]) -> dict[str, str]:
        raw_text = spec.transform(example) if spec.transform is not None else example[spec.text_field]
        if not isinstance(raw_text, str):
            raise TypeError(f"dataset {spec.path} produced non-string text: {type(raw_text).__name__}")
        return {"text": raw_text.strip()}

    return to_text


def build_utterance_text_dataset(dataset_specs: Sequence[HFStreamingTextDatasetSpec]) -> StreamingTextDataset:
    if not dataset_specs:
        raise ValueError(
            "no utterance text datasets configured; populate DEFAULT_UTTERANCE_TEXT_DATASET_SPECS "
            "in src/tts_eval/utterance_dataset_config.py"
        )

    try:
        from datasets import concatenate_datasets, load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError("datasets must be installed in the runner environment") from exc

    adapted_datasets: list[Any] = []
    for spec in dataset_specs:
        dataset = load_dataset(
            spec.path,
            name=spec.name,
            split=spec.split,
            streaming=True,
            **dict(spec.load_kwargs),
        )
        remove_columns = list(getattr(dataset, "column_names", []) or [])
        mapped = dataset.map(_build_text_mapper(spec), remove_columns=remove_columns)
        filtered = mapped.filter(lambda example: bool(example["text"]))
        adapted_datasets.append(filtered)

    merged = adapted_datasets[0] if len(adapted_datasets) == 1 else concatenate_datasets(adapted_datasets)
    return StreamingTextDataset(dataset=merged)
