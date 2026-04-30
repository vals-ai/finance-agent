from typing import Any

from model_library.base import LLMConfig, TokenRetryParams
from model_library.base.input import TextInput
from model_library.registry_utils import get_registry_model

from .get_agent import Parameters, get_agent
from .prompt import INSTRUCTIONS_PROMPT


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
    from vals.sdk.types import OutputObject  # pyright: ignore

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

    async def custom_call(test_input: str, files: dict, context: dict, question_id: str, run_id: str):
        prompt = INSTRUCTIONS_PROMPT.format(question=test_input)

        agent = get_agent(params, llm=llm)
        result = await agent.run([TextInput(text=prompt)], question_id=question_id, run_id=run_id, docent_ingest=True)

        if not result.success and result.final_error:
            print(f"\nFAIL {question_id} failed: [{result.final_error.type}] {result.final_error.message}\n")
        return OutputObject.from_agent_result(result, count_tool_metadata=True)

    return custom_call
