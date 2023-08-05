import torch
from transformers import PreTrainedModel

from openbb_chat.classifiers.abstract_zeroshot_classifier import (
    AbstractZeroshotClassifier,
)


class STransformerZeroshotClassifier(AbstractZeroshotClassifier):
    """Zero-shot classifier based on `sentence-transformers`."""

    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2", *args, **kwargs):
        """Override __init__ to set default model_id."""
        super().__init__(model_id, *args, **kwargs)

    def _compute_embed(self, inputs: dict) -> torch.Tensor:
        """Override parent method to use `sentence-transformers` models in HF."""
        return self._mean_pooling(self.model(**inputs), inputs["attention_mask"])

    def _mean_pooling(self, model_output: object, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Mean Pooling - Take attention mask into account for correct averaging.

        Args:
            model_output (`object`): output of `sentence-transformers` model in HF.
            attention_mask (`torch.Tensor`): attention mask denoting padding.

        Returns:
            `torch.Tensor`: final embedding computed from the model output.
        """
        # Code from https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
