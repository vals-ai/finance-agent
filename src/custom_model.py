import uuid
from pathlib import Path
from typing import Any

from model_library.agent import Agent, AgentConfig, AgentHooks, AgentResult, AgentTurn, truncate_oldest
from model_library.base import LLMConfig
from model_library.base.input import InputItem, TextInput
from model_library.exceptions import MaxContextWindowExceededError
from model_library.registry_utils import get_registry_model
from model_library.utils import create_file_logger
from prompt import INSTRUCTIONS_PROMPT
from tools import (
    EDGARSearch,
    ParseHtmlPage,
    RetrieveInformation,
    SubmitFinalResult,
    TavilyWebSearch,
    Tool,
)
from vals.sdk.types import OutputObject  # pyright: ignore


def agent_result_to_output_object(result: AgentResult) -> OutputObject:
    metadata = result.final_aggregated_metadata
    per_turn = [
        turn.combined_metadata.model_dump()
        for turn in result.turns
        if isinstance(turn, AgentTurn)
    ]
    return OutputObject(
        llm_output=result.final_answer,
        in_tokens=metadata.in_tokens,
        out_tokens=metadata.out_tokens,
        reasoning_tokens=metadata.reasoning_tokens,
        cache_read_tokens=metadata.cache_read_tokens,
        cache_write_tokens=metadata.cache_write_tokens,
        duration=result.final_duration_seconds,
        cost=metadata.cost.total if metadata.cost else None,
        output_context=result.model_dump(),
        metadata_per_turn=per_turn,
    )


def create_override_config(**kwargs: object) -> LLMConfig:
    valid_kwargs = {k: v for k, v in kwargs.items() if k in LLMConfig.model_fields}

    if "max_output_tokens" in kwargs and "max_tokens" not in kwargs:
        valid_kwargs["max_tokens"] = kwargs["max_output_tokens"]

    return LLMConfig(**valid_kwargs)  # pyright: ignore


async def get_custom_model(
    model_name: str,
    parameters: dict[str, Any],
    *_args: object,
    **_kwargs: object,
):
    max_turns = 50
    parameters["supports_batch"] = False
    llm = get_registry_model(model_name, create_override_config(**parameters))

    tools: list[Tool] = [
        TavilyWebSearch(),
        RetrieveInformation(llm=llm),
        ParseHtmlPage(),
        EDGARSearch(),
        SubmitFinalResult(),
    ]

    run_dir = Path("logs") / "finance_agent" / model_name.replace("/", "_") / str(uuid.uuid4())[:8]
    question_counter = [0]

    async def custom_call(test_input: str):
        question_counter[0] += 1
        question_idx = question_counter[0]
        log_file = run_dir / f"q{question_idx:03d}.log"

        with create_file_logger(f"finance_agent.q{question_idx:03d}", log_file) as logger:
            def _before_query(history: list[InputItem], last_error: Exception | None) -> list[InputItem]:
                """Only truncate on context window overflow, re-raise other errors"""
                if isinstance(last_error, MaxContextWindowExceededError):
                    return truncate_oldest(history)
                if last_error:
                    raise last_error
                return history

            agent = Agent(
                llm=llm,
                tools=tools,
                config=AgentConfig(max_turns=max_turns),
                hooks=AgentHooks(before_query=_before_query),
                logger=logger,
            )

            prompt = INSTRUCTIONS_PROMPT.format(question=test_input)
            result = await agent.run([TextInput(text=prompt)])
            return agent_result_to_output_object(result)

    return custom_call
