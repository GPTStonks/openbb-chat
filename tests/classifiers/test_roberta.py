import os
import tempfile
from unittest.mock import patch

import pytest
import torch
from transformers import AutoModel

from openbb_chat.classifiers.roberta import RoBERTaZeroshotClassifier


@patch("torch.nn.functional.normalize")
@patch("torch.cat")
@patch("torch.sum")
@patch.object(AutoModel, "from_pretrained")
def test_classify(
    mocked_automodel_frompretrained, mocked_torch_sum, mocked_torch_cat, mocked_torch_normalize
):
    mocked_torch_sum.return_value = torch.tensor([1, 0])

    roberta_zeroshot = RoBERTaZeroshotClassifier(["dog", "cat"])
    key, score, idx = roberta_zeroshot.classify("Here is a dog")

    mocked_automodel_frompretrained.assert_called_once_with("roberta-base")
    mocked_torch_sum.assert_called()
    mocked_torch_cat.assert_called()
    mocked_torch_normalize.assert_called()
    assert key == "dog"
    assert score == 1
    assert idx == 0


@patch("torch.nn.functional.normalize")
@patch("torch.cat")
@patch("torch.sum")
@patch.object(AutoModel, "from_pretrained")
def test_rank(
    mocked_automodel_frompretrained, mocked_torch_sum, mocked_torch_cat, mocked_torch_normalize
):
    mocked_torch_sum.return_value = torch.tensor([1, 0])

    roberta_zeroshot = RoBERTaZeroshotClassifier(["dog", "cat"])
    keys, scores, indices = roberta_zeroshot.rank_k("Here is a dog", k=1)

    mocked_automodel_frompretrained.assert_called_once_with("roberta-base")
    mocked_torch_sum.assert_called()
    mocked_torch_cat.assert_called()
    mocked_torch_normalize.assert_called()
    assert keys[0] == "dog"
    assert scores[0] == 1
    assert indices[0] == 0


@patch("torch.nn.functional.normalize")
@patch("torch.sum")
@patch.object(AutoModel, "from_pretrained")
def test_classify_embeddings(
    mocked_automodel_frompretrained, mocked_torch_sum, mocked_torch_normalize
):
    mocked_torch_sum.return_value = torch.tensor([1, 0])
    temp_embeds_file = os.path.join(tempfile.gettempdir(), "embeds.pt")
    torch.save(torch.randn((6, 10)), temp_embeds_file)

    roberta_zeroshot = RoBERTaZeroshotClassifier(temp_embeds_file)
    key, score, idx = roberta_zeroshot.classify("Here is a dog")

    mocked_automodel_frompretrained.assert_called_once_with("roberta-base")
    mocked_torch_sum.assert_called()
    mocked_torch_normalize.assert_called()
    assert key.shape == (10,)
    assert score == 1
    assert idx == 0
