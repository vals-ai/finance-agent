from model_library.base import LLMConfig
from model_library.registry_utils import get_registry_model
from pydantic import BaseModel

from agent import Agent
from tools import EDGARSearch, GoogleWebSearch, ParseHtmlPage, RetrieveInformation, Tool


class Parameters(BaseModel):
    model_name: str
    max_turns: int
    tools: list[str]
    llm_config: LLMConfig


def get_agent(parameters: Parameters) -> Agent:
    """Helper method to instantiate an agent with the given parameters"""
    available_tools = {
        "google_web_search": GoogleWebSearch,
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

    llm = get_registry_model(parameters.model_name, parameters.llm_config)

    agent = Agent(tools=selected_tools, llm=llm, max_turns=parameters.max_turns)

    return agent
