from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from tts_eval.utterance_texts import HFStreamingTextDatasetSpec, build_utterance_text_dataset


class FakeStreamingDataset:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.column_names = list(rows[0]) if rows else []

    def map(self, func, remove_columns=None):  # type: ignore[no-untyped-def]
        del remove_columns
        return FakeStreamingDataset([func(row) for row in self.rows])

    def filter(self, predicate):  # type: ignore[no-untyped-def]
        return FakeStreamingDataset([row for row in self.rows if predicate(row)])

    def __iter__(self):
        return iter(self.rows)


class UtteranceTextDatasetTests(unittest.TestCase):
    def test_build_dataset_concatenates_and_yields_strings(self) -> None:
        datasets_by_name = {
            "dataset_a": FakeStreamingDataset([{"body": " hello "}, {"body": "world"}]),
            "dataset_b": FakeStreamingDataset([{"text": "third"}, {"text": "   "}]),
        }

        def load_dataset(path, name=None, split=None, streaming=None, **kwargs):  # type: ignore[no-untyped-def]
            del name, split, streaming, kwargs
            return datasets_by_name[path]

        def concatenate_datasets(datasets):  # type: ignore[no-untyped-def]
            rows = []
            for dataset in datasets:
                rows.extend(list(dataset))
            return FakeStreamingDataset(rows)

        fake_datasets_module = SimpleNamespace(
            load_dataset=load_dataset,
            concatenate_datasets=concatenate_datasets,
        )

        specs = [
            HFStreamingTextDatasetSpec(path="dataset_a", text_field="body"),
            HFStreamingTextDatasetSpec(path="dataset_b"),
        ]

        with mock.patch.dict(sys.modules, {"datasets": fake_datasets_module}):
            dataset = build_utterance_text_dataset(specs)
            items = list(dataset)

        self.assertEqual(items, ["hello", "world", "third"])

    def test_empty_config_raises_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "no utterance text datasets configured"):
            build_utterance_text_dataset([])


if __name__ == "__main__":
    unittest.main()
