from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
)


class AbstractZeroshotClassifier(ABC):
    """Abstract parent class of zero-shot classifiers.

    It instantiates the necessary transformers models and loads the keys into an embedding matrix.
    """

    def __init__(
        self,
        keys: List[str],
        model_id: str,
        use_automodel_for_seq: bool = False,
        model_kwargs: dict = {},
        tokenizer_kwargs: dict = {},
    ):
        """Init method.

        Args:
            model_id (`str`):
                Name of the HF model to use.
            keys (`List[str]`):
                All possible classes, each one a short description.
            use_automodel_for_seq (`bool`):
                Whether to use `AutoModelForSequenceClassification` or `AutoModel` (default).
        """
        self.keys = keys
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
        if not use_automodel_for_seq:
            self.model = AutoModel.from_pretrained(model_id, **model_kwargs)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_id, **model_kwargs
            )
        self.model.eval()

        descr_embed_list = []
        for descr in self.keys:
            inputs = self.tokenizer(descr, return_tensors="pt")
            descr_embed = self._compute_embed(inputs)
            descr_embed_list.append(descr_embed)

        self.descr_embed = torch.nn.functional.normalize(torch.cat(descr_embed_list), dim=1)

    def classify(self, query: str) -> Tuple[str, float, int]:
        """Given a query, the most similar key is returned.

        Args:
            query (`str`):
                Short text to classify into one key.

        Returns:
            `str`: The most similar key.
            `float`: The score obtained.
            `int`: Index of the key.
        """

        inputs = self.tokenizer(query, return_tensors="pt")
        target_embed = torch.nn.functional.normalize(self._compute_embed(inputs))
        cosine_similarities = torch.sum(target_embed * self.descr_embed, dim=1)
        selected_index = torch.argmax(cosine_similarities)
        most_sim_descr = self.keys[selected_index]
        max_cosine_sim = torch.max(cosine_similarities)

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
