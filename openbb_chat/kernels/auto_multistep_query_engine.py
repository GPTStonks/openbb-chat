from typing import Awaitable, Callable, List, Optional

from llama_index.core.agent import ReActAgent
from llama_index.core.indices.query.query_transform.base import (
    StepDecomposeQueryTransform,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.query_engine import MultiStepQueryEngine
from llama_index.core.tools import FunctionTool


class SimpleLlamaIndexReActAgent(ReActAgent):
    """LlamaIndex agent based over LangChain agent tools.

    LlamaIndex agents inherit from Query Engines so they can be converted directly to other tools
    to be used by new agents (hierarchical agents).
    """

    @classmethod
    def from_scratch(
        cls,
        funcs: List[Callable],
        names: List[str],
        descriptions: List[str],
        async_funcs: Optional[List[Callable[..., Awaitable]]] = None,
        **kwargs,
    ) -> "ReActAgent":
        """Convenience constructor method from set of callables.

        Extra arguments are sent directly to `ReActAgent.from_tools`.

        Returns:
            ReActAgent
        """

        if len(funcs) == 0:
            raise ValueError("`lang_tools` cannot be empty.")
        if (
            len(funcs) != len(async_funcs or [""] * len(funcs))
            or len(funcs) != len(names)
            or len(funcs) != len(descriptions)
        ):
            raise ValueError("All lists must have the same length.")

        tools = []
        for idx, _ in enumerate(funcs):
            tools.append(
                FunctionTool.from_defaults(
                    fn=funcs[idx],
                    name=names[idx],
                    description=descriptions[idx],
                    async_fn=async_funcs[idx] if async_funcs is not None else None,
                )
            )

        return cls.from_tools(tools, **kwargs)


class AutoMultiStepQueryEngine(MultiStepQueryEngine):
    """Auto class for creating a query engine."""

    @classmethod
    def from_simple_react_agent(
        cls,
        llm: LLM,
        funcs: List[Callable],
        names: List[str],
        descriptions: List[str],
        async_funcs: Optional[List[Callable[..., Awaitable]]] = None,
        index_summary: Optional[str] = None,
        verbose: bool = False,
        step_decompose_query_transform_kwargs: dict = {},
        simple_llama_index_react_agent_kwargs: dict = {},
        **kwargs,
    ) -> "MultiStepQueryEngine":
        """Convenience constructor method from set of callables.

        Extra arguments are sent directly to `MultiStepQueryEngine` constructor.

        Returns:
            ReActAgent
        """

        # llama-index agent, inherits from query engine
        agent = SimpleLlamaIndexReActAgent.from_scratch(
            funcs=funcs,
            async_funcs=async_funcs,
            names=names,
            descriptions=descriptions,
            llm=llm,
            verbose=verbose,
            **simple_llama_index_react_agent_kwargs,
        )

        # transform to decompose input query in multiple queries
        step_decompose_transform = StepDecomposeQueryTransform(
            llm=llm, verbose=verbose, **step_decompose_query_transform_kwargs
        )

        return cls(
            query_engine=agent,
            query_transform=step_decompose_transform,
            index_summary=index_summary or "Used to answer complex questions using the Internet",
            **kwargs,
        )
