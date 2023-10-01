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
        keys: List[str] | str,
        model_id: str,
        use_automodel_for_seq: bool = False,
        model_kwargs: dict = {},
        tokenizer_kwargs: dict = {},
    ):
        """Init method.

        Args:
            keys (`List[str] | str`):
                All possible classes. There are two possibilities:
                - If a list is provided, each element is a short description to match against it.
                - If a string is provided, it is assumed to be the path to a .pt file with the embeddings of the descriptions. Shape `(num_classes, embedding_size)`.
            model_id (`str`):
                Name of the Hugging Face model to use to compute the text embeddings.
            use_automodel_for_seq (`bool`):
                Whether to use `AutoModelForSequenceClassification` (`True`) or `AutoModel` (`False`).
            model_kwargs (`dict`):
                Keyword arguments passed to Hugging Face `AutoModel.from_pretrained` or `AutoModelForSequenceClassification.from_pretrained` method.
            tokenizer_kwargs (`dict`):
                Keyword arguments passed to Hugging Face `AutoTokenizer.from_pretrained` method.
        """
        if isinstance(keys, str):
            self.keys = torch.load(keys)
        else:
            self.keys = keys
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
        if not use_automodel_for_seq:
            self.model = AutoModel.from_pretrained(model_id, **model_kwargs)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_id, **model_kwargs
            )
        self.model.eval()

        if not isinstance(self.keys, torch.Tensor):
            descr_embed_list = []
            for descr in self.keys:
                inputs = self.tokenizer(descr, return_tensors="pt")
                descr_embed = self._compute_embed(inputs)
                descr_embed_list.append(descr_embed)

            self.descr_embed = torch.nn.functional.normalize(torch.cat(descr_embed_list), dim=1)
        else:
            self.descr_embed = self.keys

    def classify(self, query: str) -> Tuple[str, float, int]:
        """Given a query, the most similar key is returned.

        Args:
            query (`str`):
                Short text to classify into one key.

        Returns:
            `str | torch.Tensor`: The most similar key. A `torch.Tensor` is returned in `keys` is initialized from a .pt file.
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

    def rank_k(self, query: str, k: int = 3) -> Tuple[List[str], List[float], List[int]]:
        """Given a query, the k most similar keys are returned, in descending order of scores.

        Args:
            query (`str`):
                Short text to classify into one key.
            k (`int`):
                No. keys to return.

        Returns:
            `List[str]`: The most similar keys.
            `List[float]`: The scores obtained.
            `List[int]`: Indices of the key.
        """
        inputs = self.tokenizer(query, return_tensors="pt")
        target_embed = torch.nn.functional.normalize(self._compute_embed(inputs))
        cosine_similarities = torch.sum(target_embed * self.descr_embed, dim=1)
        selected_scores, selected_indices = torch.topk(cosine_similarities, k=k)
        selected_scores = selected_scores.cpu().numpy().tolist()
        selected_indices = selected_indices.cpu().numpy().tolist()
        most_sim_descrs = [self.keys[selected_index] for selected_index in selected_indices]

        return most_sim_descrs, selected_scores, selected_indices

    @abstractmethod
    def _compute_embed(self, inputs: dict) -> torch.Tensor:
        """Computes the final embedding of the text given the tokenized inputs and the model.

        Args:
            inputs (`dict`): tokenized inputs to the model.

        Returns:
            `torch.Tensor`: embedding of the text. Shape (1, model_dim).
        """
        raise NotImplementedError
