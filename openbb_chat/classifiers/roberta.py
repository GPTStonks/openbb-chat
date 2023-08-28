from typing import List

import torch
from transformers import PreTrainedModel

from openbb_chat.classifiers.abstract_zeroshot_classifier import (
    AbstractZeroshotClassifier,
)


class RoBERTaZeroshotClassifier(AbstractZeroshotClassifier):
    """Zero-shot classifier based on `sentence-transformers`."""

    def __init__(self, keys: List[str], model_id: str = "roberta-base", *args, **kwargs):
        """Override __init__ to set default model_id."""
        super().__init__(keys, model_id, *args, **kwargs)

    def _compute_embed(self, inputs: dict) -> torch.Tensor:
        """Override parent method to use RoBERTa pooler output ([CLS] token)."""
        return self.model(**inputs).pooler_output
