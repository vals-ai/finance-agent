from model_library.agent import Agent, AgentConfig, AgentHooks, TurnLimit, TurnResult, default_before_query, truncate_oldest
from model_library.agent.metadata import AgentTurn, ErrorTurn, SerializableException
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

    def _before_query(history: list[InputItem], last_error: Exception | None) -> list[InputItem]:
        """Truncate on context window overflow, default behavior otherwise"""
        if isinstance(last_error, MaxContextWindowExceededError):
            return truncate_oldest(history)
        return default_before_query(history, last_error)

    def _should_stop(turn_result: TurnResult) -> bool:
        """Only stop when a done tool is called, not on text-only responses"""
        return False

    def _determine_answer(
        state: dict,
        turns: list[AgentTurn | ErrorTurn],
        final_error: SerializableException | None,
    ) -> str:
        """Only extract answer from submit_final_result tool, not from LLM text"""
        if final_error or not turns:
            return ""
        last_turn = turns[-1]
        if isinstance(last_turn, ErrorTurn):
            return ""
        done_record = next((r for r in last_turn.tool_call_records if r.tool_output.done), None)
        if done_record:
            return done_record.tool_output.output
        return ""

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
            determine_answer=_determine_answer,
        ),
    )
