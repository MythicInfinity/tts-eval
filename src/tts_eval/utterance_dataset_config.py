from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tts_eval.utterance_texts import HFStreamingTextDatasetSpec


# Populate this tuple with real Hugging Face dataset specs when the utterance
# mixture is finalized.
DEFAULT_UTTERANCE_TEXT_DATASET_SPECS: tuple["HFStreamingTextDatasetSpec", ...] = ()
