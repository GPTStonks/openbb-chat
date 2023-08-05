from unittest.mock import patch

import pytest
from transformers import AutoModel

from openbb_chat.classifiers.roberta import RoBERTaZeroshotClassifier


@patch("torch.sum")
@patch.object(AutoModel, "from_pretrained")
def test_classify(mocked_automodel_frompretrained, mocked_torch_sum):
    mocked_torch_sum.return_value = 1

    roberta_zeroshot = RoBERTaZeroshotClassifier()
    key, score, idx = roberta_zeroshot.classify("Here is a dog", ["dog", "cat"])

    mocked_automodel_frompretrained.assert_called_once_with("roberta-base")
    mocked_torch_sum.assert_called()
    assert key == "dog"
    assert score == 1
    assert idx == 0
