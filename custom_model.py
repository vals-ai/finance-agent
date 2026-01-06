import traceback

from vals.sdk.types import OutputObject
from model_library.registry_utils import get_registry_model

from .utils import create_override_config
from .agent import Agent, agent_logger
from .tools import (
    EDGARSearch,
    GoogleWebSearch,
    ParseHtmlPage,
    RetrieveInformation,
    tool_logger,
)


async def get_custom_model(
    model_name: str, parameters: dict, *args, log_level: str = "WARNING", **kwargs
):
    # set logging level
    tool_logger.setLevel(log_level)
    agent_logger.setLevel(log_level)

    max_turns = 50
    parameters["supports_batch"] = False
    llm = get_registry_model(model_name, create_override_config(**parameters))

    tools = {
        "google_web_search": GoogleWebSearch(),
        "retrieve_information": RetrieveInformation(),
        "parse_html_page": ParseHtmlPage(),
        "edgar_search": EDGARSearch(),
    }

    async def custom_call(test_input: str):
        agent = Agent(llm=llm, tools=tools, max_turns=max_turns)
        try:
            response, metadata = await agent.run(test_input)
        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"Error: {e}\n{error_traceback}")
            return {
                "llm_output": "Error when calling the agent.",
                "output_context": {"error": str(e), "traceback": error_traceback},
            }
        return OutputObject(
            llm_output=response,
            in_tokens=metadata["total_tokens"]["total_input_tokens"],
            out_tokens=metadata["total_tokens"]["total_output_tokens"],
            duration_seconds=metadata["total_duration_seconds"],
            cost=metadata["total_cost"],
            output_context=metadata,
        )

    return custom_call
