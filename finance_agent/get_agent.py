from model_library.agent import Agent, AgentConfig, AgentHooks, TurnLimit, TurnResult, default_before_query, truncate_oldest
from model_library.base import LLM, LLMConfig
from model_library.base.input import InputItem
from model_library.exceptions import MaxContextWindowExceededError
from model_library.registry_utils import get_registry_model
from pydantic import BaseModel

from .tools import (
    VALID_TOOLS,
    EDGARSearch,
    ParseHtmlPage,
    RetrieveInformation,
    SubmitFinalResult,
    TavilyWebSearch,
    Tool,
)


class Parameters(BaseModel):
    model_name: str
    max_turns: int = 50
    tools: list[str] = VALID_TOOLS
    llm_config: LLMConfig


def get_agent(
    parameters: Parameters,
    llm: LLM | None = None,
) -> Agent:
    """Helper method to instantiate an agent with the given parameters"""
    if llm is None:
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
            raise Exception(f"Tool {tool_name} not found in tools. Available tools: {available_tools.keys()}")
        tool_cls = available_tools[tool_name]
        if tool_name == "retrieve_information":
            selected_tools.append(tool_cls(llm=llm))  # type: ignore[call-arg]
        else:
            selected_tools.append(tool_cls())  # type: ignore[call-arg]

    selected_tools.append(SubmitFinalResult())

    # Agent loop stop behavior:
    #
    # The loop exits when:
    # - submit_final_result tool returns done=True -> break, no final_error
    # - max_turns exceeded -> while condition fails, final_error = MaxTurnsExceeded
    # - query error re-raised by before_query -> caught by outer except, final_error set
    # - context window exceeded -> before_query truncates history, continues (not a stop)
    # - text-only response (no tool calls) -> continues (overridden below, default would stop)
    #
    # Answer extraction (default_determine_answer):
    # - On final_error (max_turns, query error, etc): returns ""
    # - On clean exit: returns done tool output (submit_final_result)
    # - Fallback to LLM text: exists in default but is dead code here, because
    #   _should_stop=False means the only clean exit (no final_error) is the done tool break,
    #   and default_determine_answer finds the done record before reaching the text fallback.

    def _before_query(history: list[InputItem], last_error: Exception | None) -> list[InputItem]:
        """Truncate on context window overflow, re-raise all other errors (stops the loop)"""
        if isinstance(last_error, MaxContextWindowExceededError):
            return truncate_oldest(history)
        return default_before_query(history, last_error)

    def _should_stop(turn_result: TurnResult) -> bool:
        """Never stop on text-only responses — only submit_final_result or max_turns can end the loop.

        The model library default stops on text-only responses (no tool calls), but the finance agent
        should keep looping until the model calls submit_final_result or we hit max_turns.
        """
        return False

    return Agent(
        llm=llm,
        tools=selected_tools,
        name="finance",
        config=AgentConfig(
            turn_limit=TurnLimit(max_turns=parameters.max_turns),
            time_limit=None,
        ),
        hooks=AgentHooks(
            before_query=_before_query,
            should_stop=_should_stop,
        ),
    )
