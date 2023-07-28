from unittest.mock import patch

import pytest
from transformers import AutoModel, AutoTokenizer

from openbb_chat.classifiers.stransformer import STransformerZeroshotClassifier


@patch("torch.clamp")
@patch("torch.sum")
@patch.object(AutoTokenizer, "from_pretrained")
@patch.object(AutoModel, "from_pretrained")
def test_classify(
    mocked_automodel_frompretrained,
    mocked_tokenizer_frompretrained,
    mocked_torch_sum,
    mocked_torch_clamp,
):
    mocked_torch_sum.return_value = 1

    stransformer_zeroshot = STransformerZeroshotClassifier()
    key, score, idx = stransformer_zeroshot.classify("Here is a dog", ["dog", "cat"])

    mocked_automodel_frompretrained.assert_called_once_with(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    mocked_tokenizer_frompretrained.assert_called_once_with(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    mocked_torch_sum.assert_called()
    assert key == "dog"
    assert score == 1
    assert idx == 0
