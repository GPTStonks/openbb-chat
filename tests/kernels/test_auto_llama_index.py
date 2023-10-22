import os
import tempfile
from unittest.mock import patch

import pytest
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.retrievers import VectorIndexRetriever

from openbb_chat.kernels.auto_llama_index import AutoLlamaIndex


@patch.object(VectorIndexRetriever, "retrieve")
@patch.object(RetrieverQueryEngine, "query")
def test_auto_llama_index(mocked_query, mocked_retrieve):
    # load testing models
    autollamaindex = AutoLlamaIndex(
        "./docs",
        "local:sentence-transformers/all-MiniLM-L6-v2",
        "hf:sshleifer/tiny-gpt2",
        context_window=100,
        other_llama_index_response_synthesizer_kwargs={"response_mode": "simple_summarize"},
    )

    query = "What is the purpose of Index.md"

    # test retrieval
    node_list = autollamaindex.retrieve(query)
    mocked_retrieve.assert_called_once()

    # test query
    response = autollamaindex.query(query)
    mocked_query.assert_called_once()
