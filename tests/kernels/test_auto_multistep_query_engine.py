import os
from unittest.mock import patch

import pytest
from langchain_community.tools import (
    DuckDuckGoSearchResults,
    RequestsGetTool,
    WikipediaQueryRun,
)
from langchain_community.utilities import (
    DuckDuckGoSearchAPIWrapper,
    TextRequestsWrapper,
    WikipediaAPIWrapper,
)
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.llms.openai import OpenAI

from openbb_chat.kernels import AutoMultiStepQueryEngine


@pytest.mark.asyncio
@patch.object(BaseQueryEngine, "query")
@patch.object(BaseQueryEngine, "aquery")
async def test_factory_react_agent(mocked_aquery, mocked_query):
    os.environ["OPENAI_API_KEY"] = "sk-..."

    # LangChain BaseTools to use
    search_tool = DuckDuckGoSearchResults(api_wrapper=DuckDuckGoSearchAPIWrapper())
    wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    request_get_tool = RequestsGetTool(requests_wrapper=TextRequestsWrapper())

    # base parameters for the auto model
    def search_tool_func(x):
        return search_tool.run(x)

    def wikipedia_tool_func(x):
        return wikipedia_tool.run(x)

    def request_get_tool_func(x):
        return request_get_tool.run(x)

    async def search_tool_async_func(x):
        return await search_tool.arun(x)

    async def wikipedia_tool_async_func(x):
        return await wikipedia_tool.arun(x)

    async def request_get_tool_async_func(x):
        return await request_get_tool.arun(x)

    # create query engine using factory
    query_engine = AutoMultiStepQueryEngine.from_simple_react_agent(
        llm=OpenAI(model="gpt-4-0125-preview"),
        funcs=[search_tool_func, wikipedia_tool_func, request_get_tool_func],
        async_funcs=[
            search_tool_async_func,
            wikipedia_tool_async_func,
            request_get_tool_async_func,
        ],
        names=[search_tool.name, wikipedia_tool.name, request_get_tool.name],
        descriptions=[
            search_tool.description,
            wikipedia_tool.description,
            request_get_tool.description,
        ],
        verbose=True,
        index_summary="Useful to get information on the Internet",
    )
    query_engine.query("Whatever")
    await query_engine.aquery("Whatever again")

    assert len(query_engine.get_prompts()) == 3
    mocked_aquery.assert_called_once()
    mocked_query.assert_called_once()


@pytest.mark.asyncio
@patch.object(BaseQueryEngine, "query")
@patch.object(BaseQueryEngine, "aquery")
async def test_factory_openai_agent(mocked_aquery, mocked_query):
    os.environ["OPENAI_API_KEY"] = "sk-..."

    # LangChain BaseTools to use
    search_tool = DuckDuckGoSearchResults(api_wrapper=DuckDuckGoSearchAPIWrapper())
    wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    request_get_tool = RequestsGetTool(requests_wrapper=TextRequestsWrapper())

    # base parameters for the auto model
    def search_tool_func(x):
        return search_tool.run(x)

    def wikipedia_tool_func(x):
        return wikipedia_tool.run(x)

    def request_get_tool_func(x):
        return request_get_tool.run(x)

    async def search_tool_async_func(x):
        return await search_tool.arun(x)

    async def wikipedia_tool_async_func(x):
        return await wikipedia_tool.arun(x)

    async def request_get_tool_async_func(x):
        return await request_get_tool.arun(x)

    # create query engine using factory
    query_engine = AutoMultiStepQueryEngine.from_simple_openai_agent(
        llm=OpenAI(model="gpt-4-0125-preview"),
        funcs=[search_tool_func, wikipedia_tool_func, request_get_tool_func],
        async_funcs=[
            search_tool_async_func,
            wikipedia_tool_async_func,
            request_get_tool_async_func,
        ],
        names=[search_tool.name, wikipedia_tool.name, request_get_tool.name],
        descriptions=[
            search_tool.description,
            wikipedia_tool.description,
            request_get_tool.description,
        ],
        verbose=True,
        index_summary="Useful to get information on the Internet",
    )
    query_engine.query("Whatever")
    await query_engine.aquery("Whatever again")

    assert len(query_engine.get_prompts()) == 3
    mocked_aquery.assert_called_once()
    mocked_query.assert_called_once()
