import logging

from model_library.agent import Agent, AgentConfig, AgentHooks, default_before_query, truncate_oldest
from model_library.base import LLMConfig
from model_library.base.input import InputItem
from model_library.exceptions import MaxContextWindowExceededError
from model_library.registry_utils import get_registry_model
from pydantic import BaseModel
from tools import (
    EDGARSearch,
    ParseHtmlPage,
    RetrieveInformation,
    SubmitFinalResult,
    TavilyWebSearch,
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
) -> Agent:
    """Helper method to instantiate an agent with the given parameters"""
    llm = get_registry_model(parameters.model_name, parameters.llm_config)

    available_tools: dict[str, type[Tool]] = {
        "web_search": TavilyWebSearch,
        "retrieve_information": RetrieveInformation,
        "parse_html_page": ParseHtmlPage,
        "edgar_search": EDGARSearch,
    }

    selected_tools: list[Tool] = []
    for tool_name in parameters.tools:
        if tool_name not in available_tools:
            raise Exception(
                f"Tool {tool_name} not found in tools. Available tools: {available_tools.keys()}"
            )
        tool_cls = available_tools[tool_name]
        if tool_name == "retrieve_information":
            selected_tools.append(tool_cls(llm=llm))  # type: ignore[call-arg]
        else:
            selected_tools.append(tool_cls())  # type: ignore[call-arg]

    selected_tools.append(SubmitFinalResult())

    def _before_query(history: list[InputItem], last_error: Exception | None) -> list[InputItem]:
        """Truncate on context window overflow, default behavior otherwise"""
        if isinstance(last_error, MaxContextWindowExceededError):
            return truncate_oldest(history)
        return default_before_query(history, last_error)

    return Agent(
        llm=llm,
        tools=selected_tools,
        config=AgentConfig(max_turns=parameters.max_turns),
        hooks=AgentHooks(before_query=_before_query),
        logger=logging.getLogger(logger_name or f"agent.{parameters.model_name}"),
    )
