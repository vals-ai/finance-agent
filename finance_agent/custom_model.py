from typing import Any

from model_library.agent import AgentResult
from model_library.base import LLMConfig, TokenRetryParams
from model_library.base.input import TextInput
from model_library.registry_utils import get_registry_model
from vals.sdk.types import OutputObject  # pyright: ignore

from finance_agent.get_agent import Parameters, get_agent
from finance_agent.prompt import INSTRUCTIONS_PROMPT


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

    llm = get_registry_model(model_name, params.llm_config)

    token_retry_params = parameters.get("token_retry_params", None)
    if token_retry_params:
        await llm.init_token_retry(
            token_retry_params=TokenRetryParams.model_validate(token_retry_params),
        )

    question_counter = 0

    async def custom_call(test_input: str):
        nonlocal question_counter
        question_counter += 1
        question_idx = question_counter

        prompt = INSTRUCTIONS_PROMPT.format(question=test_input)

        agent = get_agent(params, llm=llm)
        result = await agent.run([TextInput(text=prompt)], question_id=f"q{question_idx:03d}")

        if not result.success and result.final_error:
            print(f"\nFAIL Question {question_idx} failed: [{result.final_error.type}] {result.final_error.message}\n")
        return agent_result_to_output_object(result)

    return custom_call
