from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
from rich.progress import track
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
)


class AbstractZeroshotClassifier(ABC):
    """Abstract parent class of zero-shot classifiers.

    It instantiates the necessary transformers models.
    """

    def __init__(
        self,
        model_id: str,
        use_automodel_for_seq: bool = False,
        model_kwargs: dict = {},
        tokenizer_kwargs: dict = {},
    ):
        """Init method.

        Args:
            model_id (`str`):
                Name of the HF model to use.
            model_type (`bool`):
                Whether to use `AutoModelForSequenceClassification` or `AutoModel` (default).
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)

        if not use_automodel_for_seq:
            self.model = AutoModel.from_pretrained(model_id, **model_kwargs)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_id, **model_kwargs
            )
        self.model.eval()

    def classify(self, query: str, keys: List[str]) -> Tuple[str, float, int]:
        """Given a query, the most similar key is returned.

        Args:
            query (`str`):
                Short text to classify into one key.
            keys (`List[str]`):
                All possible classes, each one a short description.

        Returns:
            `str`: The most similar key.
            `float`: The score obtained.
            `int`: Index of the key.
        """

        inputs = self.tokenizer(query, return_tensors="pt")
        target_embed = self._compute_embed(inputs)

        max_cosine_sim = -1  # min. possible cosine similarity
        most_sim_descr = ""
        selected_index = -1
        for idx, descr in track(enumerate(keys), total=len(keys), description="Processing..."):
            inputs = self.tokenizer(descr, return_tensors="pt")
            descr_embed = self._compute_embed(inputs)

            cosine_sim = torch.sum(
                torch.nn.functional.normalize(target_embed)
                * torch.nn.functional.normalize(descr_embed)
            )
            if cosine_sim > max_cosine_sim:
                most_sim_descr = descr
                max_cosine_sim = cosine_sim
                selected_index = idx

        return most_sim_descr, max_cosine_sim, selected_index

    @abstractmethod
    def _compute_embed(self, inputs: dict) -> torch.Tensor:
        """Computes the final embedding of the text given the tokenized inputs and the model.

        Args:
            inputs (`dict`): tokenized inputs to the model.

        Returns:
            `torch.Tensor`: embedding of the text. Shape (1, model_dim).
        """
        raise NotImplementedError
