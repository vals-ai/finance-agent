import traceback
from typing import Any

from agent import Agent, agent_logger
from logger import setup_question_logging, teardown_question_logging
from model_library.base import LLMConfig
from model_library.registry_utils import get_registry_model
from run_agent import create_question_directory, create_run_directory
from tools import (
    EDGARSearch,
    ParseHtmlPage,
    RetrieveInformation,
    SubmitFinalResult,
    TavilyWebSearch,
    Tool,
    tool_logger,
)
from vals.sdk.types import OutputObject  # pyright: ignore


def create_override_config(**kwargs: object) -> LLMConfig:
    # Filter kwargs to only include valid LLMConfig fields
    valid_kwargs = {k: v for k, v in kwargs.items() if k in LLMConfig.model_fields}

    # hardcode fix for max output tokens
    if "max_output_tokens" in kwargs and "max_tokens" not in kwargs:
        valid_kwargs["max_tokens"] = kwargs["max_output_tokens"]

    return LLMConfig(**valid_kwargs)  # pyright: ignore


async def get_custom_model(
    model_name: str,
    parameters: dict[str, Any],
    log_level: str = "WARNING",
    *_args: object,
    **_kwargs: object,
):
    # set logging level
    tool_logger.setLevel(log_level)
    agent_logger.setLevel(log_level)

    max_turns = 50
    parameters["supports_batch"] = False
    llm = get_registry_model(model_name, create_override_config(**parameters))

    tools: dict[str, Tool] = {
        "web_search": TavilyWebSearch(),
        "retrieve_information": RetrieveInformation(),
        "parse_html_page": ParseHtmlPage(),
        "edgar_search": EDGARSearch(),
        "submit_final_result": SubmitFinalResult(),
    }

    # Create run directory for logging
    run_dir = create_run_directory(model_name)
    question_counter = [0]  # Use list to allow mutation in closure

    async def custom_call(test_input: str):
        # NOTE: cannot reuse agent as it keeps track of self.messages
        agent = Agent(llm=llm, tools=tools, max_turns=max_turns)

        # Create question directory and setup logging
        question_counter[0] += 1
        question_dir = create_question_directory(run_dir, question_counter[0])
        setup_question_logging(question_dir, ["agent", "tools"])

        try:
            response, metadata = await agent.run(test_input, question_dir=question_dir)
        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"Error: {e}\n{error_traceback}")
            return {
                "llm_output": "Error when calling the agent.",
                "output_context": {"error": str(e), "traceback": error_traceback},
            }
        finally:
            teardown_question_logging(question_dir)

        return OutputObject(
            llm_output=response,
            in_tokens=metadata.total_tokens.total_input_tokens,
            out_tokens=metadata.total_tokens.total_output_tokens,
            duration_seconds=metadata.total_duration_seconds,
            cost=metadata.total_cost,
            output_context=metadata.model_dump(),
        )

    return custom_call
