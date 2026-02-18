import json
import os
import traceback
import uuid
from abc import ABC
from datetime import datetime
from typing import Any

from model_library.base import (
    LLM,
    InputItem,
    QueryResult,
    QueryResultMetadata,
    RawResponse,
    TextInput,
    ToolCall,
    ToolResult,
)
from model_library.exceptions import MaxContextWindowExceededError
from pydantic import BaseModel

from logger import get_logger
from prompt import INSTRUCTIONS_PROMPT
from tools import Tool
from utils import _merge_statistics

agent_logger = get_logger(__name__)


def dict_replace_none_with_zero(d: dict) -> dict:
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = dict_replace_none_with_zero(v)
        else:
            result[k] = 0 if v is None else v
    return result


class ModelException(Exception):
    """
    Raised on model errors - not retried by default
    """

    pass


class TurnMetadata(BaseModel):
    tool_calls: list[dict[str, Any]] = []
    errors: list[str] = []
    query_metadata: QueryResultMetadata
    retrieval_metadata: QueryResultMetadata
    combined_metadata: QueryResultMetadata
    total_cost: float


class Metadata(BaseModel):
    session_id: str
    model_key: str
    user_input: str

    start_time: str = datetime.now().isoformat()
    end_time: str | None = None
    total_duration_seconds: float | None = None

    total_tokens: QueryResultMetadata = QueryResultMetadata()
    total_tokens_retrieval: QueryResultMetadata = QueryResultMetadata()
    total_tokens_query: QueryResultMetadata = QueryResultMetadata()
    total_cost: float = 0

    tool_usage: dict[str, int] = {}
    tool_calls_count: int = 0

    api_calls_count: int = 0

    turns: list[TurnMetadata] = []
    error_count: int = 0

    final_answer: str | None = None


class Agent(ABC):
    def __init__(
        self,
        tools: dict[str, Tool],
        llm: LLM,
        max_turns: int = 20,
        instructions_prompt: str = INSTRUCTIONS_PROMPT,
        logger_name: str | None = None,
        tools_logger_name: str | None = None,
    ):
        self.tools: dict[str, Tool] = tools
        self.llm: LLM = llm
        self.max_turns: int = max_turns
        self.instructions_prompt: str = instructions_prompt

        self.logger = get_logger(logger_name) if logger_name else agent_logger
        self.tools_logger = get_logger(tools_logger_name) if tools_logger_name else None
        # NOTE: Don't set self.llm.logger here - the LLM is shared across agents.
        # Instead, pass query_logger to each llm.query() call.

    async def _process_tool_calls(
        self,
        tool_calls: list[ToolCall],
        data_storage: dict[str, Any],
        turn_metadata: TurnMetadata,
    ):
        """
        Helper method to process tool calls, handling errors, validating arguments,
        and generating the results.
        """

        tool_results: list[ToolResult] = []
        tool_call_metadatas: list[dict[str, Any]] = []
        errors: list[str] = []

        for tool_call in tool_calls:
            tool_name = tool_call.name

            arguments = tool_call.args
            tool_call_metadata = {
                "tool_name": tool_name,
                "arguments": arguments,
                "success": False,
                "error": None,
            }

            if tool_name not in self.tools:
                error_msg = f"Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}"

                tool_call_metadata["error"] = error_msg
                tool_call_metadatas.append(tool_call_metadata)
                turn_metadata.errors.append(error_msg)

                tool_result = ToolResult(tool_call=tool_call, result=error_msg)
                tool_results.append(tool_result)
                continue

            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    error_msg = f"Tool call arguments were not valid json: {arguments}"

                    tool_call_metadata["error"] = error_msg
                    tool_call_metadatas.append(tool_call_metadata)
                    errors.append(error_msg)

                    tool_result = ToolResult(tool_call=tool_call, result=error_msg)
                    tool_results.append(tool_result)
                    continue

            raw_tool_result = await self.tools[tool_name](
                arguments, data_storage, self.llm, self.tools_logger
            )

            if tool_name == "retrieve_information":
                if "usage" in raw_tool_result:
                    tool_token_usage: QueryResultMetadata = raw_tool_result["usage"]
                    turn_metadata.retrieval_metadata = tool_token_usage
                    turn_metadata.combined_metadata += tool_token_usage
                    turn_metadata.total_cost += (
                        tool_token_usage.cost.total if tool_token_usage.cost else 0
                    )

            if raw_tool_result["success"]:
                tool_call_metadata["success"] = True
            else:
                tool_call_metadata["error"] = raw_tool_result["result"]
                errors.append(raw_tool_result["result"])

            tool_results.append(
                ToolResult(tool_call=tool_call, result=raw_tool_result["result"])
            )

            tool_call_metadatas.append(tool_call_metadata)

        turn_metadata.tool_calls.extend(tool_call_metadatas)

        return tool_results

    def _shorten_message_history(self):
        """
        When the max context of the agent is exceeded, we remove some of the earliest messages to
        free up space.

        We always leave the first input, and from there, remove begin removing model responses
        and associated tool results.

        NOTE: This function is very rarely called, most models are able to complete the task within the context window.
        """
        self.logger.warning(
            "Max Context Window Exceeded. "
            "Removing first model response from the stack, "
            "as well as all associated tool calls and results."
        )

        # Remove all response items from the first model call - certain models
        # return multiple list items per call
        removed_count = 0
        while len(self.messages) > 1 and isinstance(self.messages[1], RawResponse):
            self.messages.pop(1)
            removed_count += 1

        self.logger.info(f"Removed {removed_count} model response item(s)")

        # Remove all input items. 99% of the time, this will just be ToolResults
        # from the previous batch of inputs, but we need to remove all input items,
        # otherwise we may get stuck.
        input_item_count = 0
        while len(self.messages) > 1 and not isinstance(self.messages[1], RawResponse):
            self.messages.pop(1)
            input_item_count += 1

        if input_item_count > 0:
            self.logger.info(f"Removed {input_item_count} InputItem(s)")

    async def _process_turn(
        self, turn_count: int, data_storage: dict[str, Any]
    ) -> tuple[str | None, TurnMetadata]:
        """
        Process a single turn in the agent's conversation.

        Args:
            turn_count (int): The current turn number
            data_storage (dict): Storage for conversation data

        Returns:
            tuple: (final_answer, turn_metadata, should_continue)
        """
        self.logger.info(f"\033[1;34m[TURN {turn_count}]\033[0m")

        tool_definitions = [tool.tool_definition for tool in self.tools.values()]
        self.logger.info(
            f"\033[1;35m[TOOLS AVAILABLE]\033[0m {[tool.name for tool in tool_definitions]}"
        )

        try:
            response: QueryResult = await self.llm.query(
                input=self.messages, tools=tool_definitions, query_logger=self.logger
            )
        except MaxContextWindowExceededError:
            raise
        except Exception as e:
            self.logger.critical(f"Error: {e}")
            self.logger.critical(f"Traceback: {traceback.format_exc()}")
            raise ModelException(e)

        self.messages = response.history

        response_text = response.output_text
        reasoning_text = response.reasoning
        tool_calls: list[ToolCall] = response.tool_calls

        self.logger.info(
            f"\033[1;36m[TOOL CALLS RECEIVED]\033[0m {len(tool_calls)} tool calls: {[tc.name for tc in tool_calls]}"
        )

        if not response.metadata.cost:
            raise Exception("LLM response metadata has no cost")

        turn_metadata = TurnMetadata(
            query_metadata=response.metadata,
            retrieval_metadata=QueryResultMetadata(),
            combined_metadata=response.metadata,
            total_cost=response.metadata.cost.total,
        )

        if reasoning_text:
            self.logger.info(f"\033[1;33m[LLM REASONING]\033[0m {reasoning_text}")
        if response_text:
            self.logger.info(f"\033[1;33m[LLM RESPONSE]\033[0m {response_text}")

        if tool_calls:
            tool_results = await self._process_tool_calls(
                tool_calls, data_storage, turn_metadata
            )
            self.messages.extend(tool_results)

            submit_final_result_tool_result = next(
                (
                    tool_result
                    for tool_result in tool_results
                    if tool_result.tool_call.name == "submit_final_result"
                ),
                None,
            )
            if submit_final_result_tool_result:
                final_answer = json.loads(submit_final_result_tool_result.result)[
                    "result"
                ]
                self.logger.info(f"\033[1;32m[FINAL ANSWER]\033[0m {final_answer}")
                return final_answer, turn_metadata

        return None, turn_metadata

    async def run(
        self,
        question: str,
        question_dir: str,
        session_id: str | None = None,
    ) -> tuple[str, Metadata]:
        """
        Run the agent on a question from the user.

        Args:
            question (str): The user's question
            session_id (str, optional): A unique identifier for this session

        Returns:
            tuple[str, dict]: The final answer and metadata about the run
        """
        session_id = session_id or str(uuid.uuid4())

        assert self.llm._registry_key

        metadata = Metadata(
            session_id=session_id,
            model_key=self.llm._registry_key,
            user_input=question,
        )

        data_storage = {}

        initial_prompt = self.instructions_prompt.format(question=question)

        initial_message = TextInput(text=initial_prompt)
        self.messages: list[InputItem] = [initial_message]

        self.logger.info(f"\033[1;34m[USER INSTRUCTIONS]\033[0m {initial_prompt}")

        turn_count = 0

        # setting either breaks control loop
        final_answer = None
        breaking_error = None

        while turn_count < self.max_turns and breaking_error is not None and final_answer is not None:
            turn_count += 1

            try:
                final_answer, turn_metadata = await self._process_turn(
                    turn_count, data_storage
                )

                metadata.turns.append(turn_metadata)

            except MaxContextWindowExceededError:
                self._shorten_message_history()  # TODO
            except ModelException as e:
                result = f"Model exception occurred: {e}"
                metadata.error_count += 1
                self.logger.error(result)

                # breaks out of turn loop
                breaking_error = e

            except Exception as e:
                metadata.error_count += 1
                self.logger.error(f"\033[1;31m[ERROR]\033[0m {e}")
                self.logger.error(
                    f"\033[1;31m[traceback]\033[0m {traceback.format_exc()}"
                )

                error_message = TextInput(
                    text=f"An error occurred: {e}. Please review what happened and try a different approach."
                )
                self.messages.append(error_message)
                
                # breaks out of turn loop
                breaking_error = e

        metadata.end_time = datetime.now().isoformat()

        if final_answer:
            metadata.final_answer = final_answer

        metadata = _merge_statistics(metadata)

        if question_dir:
            log_path = os.path.join(question_dir, "trajectory.json")
            with open(log_path, "w") as f:
                json.dump(metadata.model_dump(), f, indent=2)

        if final_answer:
            return final_answer, metadata
        elif turn_count >= self.max_turns:
            return "Max turns reached without final answer.", metadata
        elif breaking_error:
            return f"Unable to generate answer due to the following error: {breaking_error}", metadata
        else:
            # this should never happen
            return "Unable to generate answer for unknown reason", metadata
