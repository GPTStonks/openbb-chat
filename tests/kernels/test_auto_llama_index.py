import os
import tempfile
from unittest.mock import patch

import pytest
from llama_index import (
    ServiceContext,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.llms import HuggingFaceLLM
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


@patch.object(VectorIndexRetriever, "retrieve")
@patch.object(RetrieverQueryEngine, "query")
def test_auto_llama_index_persistance(mocked_query, mocked_retrieve):
    # persist an index to test loading
    docs_sdk = SimpleDirectoryReader("./docs", recursive=True).load_data()

    # service context to customize the models used by LlamaIndex
    service_context = ServiceContext.from_defaults(
        embed_model="local:sentence-transformers/all-MiniLM-L6-v2",
        llm=HuggingFaceLLM(
            tokenizer_name="sshleifer/tiny-gpt2",
            model_name="sshleifer/tiny-gpt2",
        ),
    )
    nodes_sdk = service_context.node_parser.get_nodes_from_documents(docs_sdk)

    # initialize storage context (by default it's in-memory)
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes_sdk)

    # create vector store index
    index = VectorStoreIndex(
        nodes_sdk,
        show_progress=True,
        service_context=service_context,
        storage_context=storage_context,
    )
    index.storage_context.persist(persist_dir="./st")

    # load testing models
    autollamaindex = AutoLlamaIndex(
        "vsi:./st",
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


@patch.object(VectorIndexRetriever, "retrieve")
@patch.object(RetrieverQueryEngine, "query")
def test_auto_llama_index_query_with_model(mocked_query, mocked_retrieve):
    # load testing models
    autollamaindex = AutoLlamaIndex(
        "./docs",
        "local:sentence-transformers/all-MiniLM-L6-v2",
        "hf:sshleifer/tiny-gpt2",
        context_window=100,
        other_llama_index_response_synthesizer_kwargs={"response_mode": "simple_summarize"},
    )

    query = "What is the purpose of Index.md"

    # test query
    response = autollamaindex.query_with_model(query, "hf:sshleifer/tiny-gpt2")
    mocked_query.assert_called_once()


@patch.object(VectorIndexRetriever, "retrieve")
@patch.object(RetrieverQueryEngine, "synthesize")
def test_auto_llama_index_synth(mocked_synthesize, mocked_retrieve):
    # load testing models
    autollamaindex = AutoLlamaIndex(
        "./docs",
        "local:sentence-transformers/all-MiniLM-L6-v2",
        "hf:sshleifer/tiny-gpt2",
        context_window=100,
        other_llama_index_response_synthesizer_kwargs={"response_mode": "simple_summarize"},
    )

    query = "What is the purpose of Index.md"

    # test query
    node_list = autollamaindex.retrieve(query)
    response = autollamaindex.synth(query, node_list)
    mocked_synthesize.assert_called_once()
