from model_library.base import LLMConfig
from model_library.registry_utils import get_registry_model
from pydantic import BaseModel

from agent import Agent
from tools import (
    EDGARSearch,
    TavilyWebSearch,
    ParseHtmlPage,
    RetrieveInformation,
    SubmitFinalResult,
    Tool,
    VALID_TOOLS,
)


class Parameters(BaseModel):
    model_name: str
    max_turns: int = 50
    tools: list[str] = VALID_TOOLS
    llm_config: LLMConfig


def get_agent(
    parameters: Parameters,
    logger_name: str | None = None,
    tools_logger_name: str | None = None,
) -> Agent:
    """Helper method to instantiate an agent with the given parameters"""
    available_tools = {
        "web_search": TavilyWebSearch,
        "retrieve_information": RetrieveInformation,
        "parse_html_page": ParseHtmlPage,
        "edgar_search": EDGARSearch,
    }

    selected_tools: dict[str, Tool] = {}
    for tool in parameters.tools:
        if tool not in available_tools:
            raise Exception(
                f"Tool {tool} not found in tools. Available tools: {available_tools.keys()}"
            )
        selected_tools[tool] = available_tools[tool]()

    selected_tools["submit_final_result"] = SubmitFinalResult()

    llm = get_registry_model(parameters.model_name, parameters.llm_config)

    agent = Agent(
        tools=selected_tools,
        llm=llm,
        max_turns=parameters.max_turns,
        logger_name=logger_name,
        tools_logger_name=tools_logger_name,
    )

    return agent
