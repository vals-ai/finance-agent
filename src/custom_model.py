from typing import Any

from get_agent import Parameters, get_agent
from model_library.agent import AgentResult
from model_library.base import LLMConfig
from model_library.base.input import TextInput
from model_library.utils import create_file_logger, create_run_dir
from prompt import INSTRUCTIONS_PROMPT
from vals.sdk.types import OutputObject  # pyright: ignore


def agent_result_to_output_object(result: AgentResult) -> OutputObject:
    metadata = result.final_aggregated_metadata
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
        error=str(result.final_error) if result.final_error else None,
    )


def create_override_config(**kwargs: object) -> LLMConfig:
    if "max_output_tokens" in kwargs and "max_tokens" not in kwargs:
        kwargs["max_tokens"] = kwargs["max_output_tokens"]

    return LLMConfig.model_validate(kwargs, strict=False)


async def get_custom_model(
    model_name: str,
    parameters: dict[str, Any],
    *_args: object,
    **_kwargs: object,
):
    params = Parameters(
        model_name=model_name,
        llm_config=create_override_config(**parameters),
    )

    run_dir = create_run_dir("finance_agent", model_name)
    question_counter = 0

    async def custom_call(test_input: str):
        nonlocal question_counter
        question_counter += 1
        question_idx = question_counter
        log_file = run_dir / f"q{question_idx:03d}.log"

        with create_file_logger(f"finance_agent.q{question_idx:03d}", log_file) as logger:
            agent = get_agent(params, logger_name=logger.name)

            prompt = INSTRUCTIONS_PROMPT.format(question=test_input)
            result = await agent.run([TextInput(text=prompt)])
            return agent_result_to_output_object(result)

    return custom_call
