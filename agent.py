import json
import os
import re
import traceback
import uuid
from abc import ABC
from datetime import datetime
from collections import defaultdict

from model_library.base import (
    LLM,
    ToolCall,
    ToolResult,
    QueryResult,
    InputItem,
    TextInput,
)
from model_library.exceptions import MaxContextWindowExceededError

from logger import get_logger
from tools import Tool
from utils import INSTRUCTIONS_PROMPT, _merge_statistics, TOKEN_KEYS, COST_KEYS

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


class Agent(ABC):
    def __init__(
        self,
        tools: dict[str, Tool],
        llm: LLM,
        max_turns: int = 20,
        instructions_prompt: str = INSTRUCTIONS_PROMPT,
    ):
        self.tools = tools
        self.llm = llm
        self.max_turns = max_turns
        self.instructions_prompt = instructions_prompt

    async def _find_final_answer(self, response_text: str) -> str:
        """
        Search through the response text for the presence of 'FINAL ANSWER:', and if its present,
        extract the answer and any sources
        """
        final_answer_pattern = re.compile(r"FINAL ANSWER:", re.IGNORECASE)

        if isinstance(response_text, str) and final_answer_pattern.search(
            response_text
        ):
            final_answer_match = re.search(
                r"FINAL ANSWER:(.*?)(?:\{\"sources\"|\Z)",
                response_text,
                re.DOTALL,
            )
            sources_match = re.search(r"(\{\"sources\".*\})", response_text, re.DOTALL)

            answer_text = (
                final_answer_match.group(1).strip() if final_answer_match else ""
            )

            sources_text = sources_match.group(1) if sources_match else ""

            final_answer = answer_text
            if sources_text:
                final_answer = f"{answer_text}\n\n{sources_text}"

            agent_logger.info(f"\033[1;32m[FINAL ANSWER]\033[0m {final_answer}")
            return final_answer

        return None

    async def _process_tool_calls(
        self, tool_calls: list[ToolCall], data_storage: dict, turn_metadata: dict
    ):
        """
        Helper method to process tool calls, handling errors, validating arguments,
        and generating the results.
        """

        tool_results: list[ToolResult] = []
        tool_call_metadatas: list[dict] = []
        errors: list[str] = []

        for tool_call in tool_calls:
            tool_name = tool_call.name

            # unpacks tool call arguments
            arguments = tool_call.args
            tool_call_metadata = {
                "tool_name": tool_name,
                "arguments": arguments,
                "success": False,
                "error": None,
            }

            # Validate tool_name exists
            if tool_name not in self.tools:
                error_msg = f"Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}"

                tool_call_metadata["error"] = error_msg
                tool_call_metadatas.append(tool_call_metadata)
                turn_metadata["errors"].append(error_msg)

                tool_result = ToolResult(tool_call=tool_call, result=error_msg)
                tool_results.append(tool_result)
                continue

            # Validate tool arguments are JSON-parseable
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

            # Call tools with appropriate arguments
            if tool_name == "retrieve_information":
                raw_tool_result = await self.tools[tool_name](
                    arguments, data_storage, self.llm
                )
                if "usage" in raw_tool_result:
                    # Retrieval can use LLM tokens too, so we need to track them here
                    tool_token_usage = raw_tool_result["usage"]
                    turn_metadata["retrieval_metadata"] = {**tool_token_usage}
                    for key in TOKEN_KEYS:
                        turn_metadata["combined_metadata"][key] += (
                            tool_token_usage.get(key, 0) or 0
                        )
                    for key in COST_KEYS:
                        turn_metadata["combined_metadata"]["cost"][key] += (
                            tool_token_usage.get("cost", {}).get(key, 0) or 0
                        )
                    turn_metadata["total_cost"] += tool_token_usage["cost"]["total"]

            elif tool_name == "parse_html_page":
                raw_tool_result = await self.tools[tool_name](arguments, data_storage)
            else:
                raw_tool_result = await self.tools[tool_name](arguments)

            if raw_tool_result["success"]:
                # Add tool result to messages
                tool_call_metadata["success"] = True
            else:
                tool_call_metadata["error"] = raw_tool_result["result"]
                errors.append(raw_tool_result["result"])

            tool_result = ToolResult(
                tool_call=tool_call, result=raw_tool_result["result"]
            )
            tool_results.append(tool_result)

            tool_call_metadatas.append(tool_call_metadata)

        turn_metadata["tool_calls"].extend(tool_call_metadatas)

        return tool_results

    async def _process_turn(self, turn_count, data_storage):
        """
        Process a single turn in the agent's conversation.

        Args:
            turn_count (int): The current turn number
            data_storage (dict): Storage for conversation data

        Returns:
            tuple: (final_answer, turn_metadata, should_continue)
        """
        agent_logger.info(f"\033[1;34m[TURN {turn_count}]\033[0m")

        tool_definitions = [tool.get_tool_definition() for tool in self.tools.values()]
        agent_logger.info(
            f"\033[1;35m[TOOLS AVAILABLE]\033[0m {[tool.name for tool in tool_definitions]}"
        )

        try:
            response: QueryResult = await self.llm.query(
                input=self.messages, tools=tool_definitions
            )
        except Exception as e:
            agent_logger.critical(f"Error: {e}")
            agent_logger.critical(f"Traceback: {traceback.format_exc()}")
            raise ModelException(e)

        self.messages = response.history

        response_text = response.output_text
        reasoning_text = response.reasoning
        tool_calls: list[ToolCall] = response.tool_calls

        agent_logger.info(
            f"\033[1;36m[TOOL CALLS RECEIVED]\033[0m {len(tool_calls)} tool calls: {[tc.name for tc in tool_calls]}"
        )

        turn_metadata = {
            "tool_calls": [],
            "errors": [],
            # Metadata for original LLM query for this turn
            "query_metadata": dict_replace_none_with_zero(
                response.metadata.model_dump()
            ),
            # Metadata from LLM usage by the 'retrieve_information' tool
            "retrieval_metadata": defaultdict(int),
            # Metadata from combined LLM query and tool calls for this turn
            "combined_metadata": dict_replace_none_with_zero(
                response.metadata.model_dump()
            ),
            "total_cost": response.metadata.cost.total,
        }

        # Log the thinking content if available
        if reasoning_text:
            agent_logger.info(f"\033[1;33m[LLM REASONING]\033[0m {reasoning_text}")

        if response_text:
            agent_logger.info(f"\033[1;33m[LLM RESPONSE]\033[0m {response_text}")

        if tool_calls:
            tool_results = await self._process_tool_calls(
                tool_calls, data_storage, turn_metadata
            )
            self.messages.extend(tool_results)

        else:
            # Look for the text "FINAL ANSWER:" in the response text
            final_answer = await self._find_final_answer(response_text)

            if final_answer:
                return final_answer, turn_metadata, False

        return None, turn_metadata, True

    async def run(self, question: str, session_id: str = None) -> tuple[str, dict]:
        """
        Run the agent on a question from the user.

        Args:
            question (str): The user's question
            session_id (str, optional): A unique identifier for this session

        Returns:
            tuple[str, dict]: The final answer and metadata about the run
        """
        # Initialize metadata
        session_id = session_id or str(uuid.uuid4())
        metadata = {
            "session_id": session_id,
            "user_input": question,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "total_duration_seconds": 0,
            "total_tokens": defaultdict(int),
            "total_tokens_retrieval": defaultdict(int),
            "total_tokens_query": defaultdict(int),
            "turns": [],
            "tool_usage": {},
            "tool_calls_count": 0,
            "api_calls_count": 0,
            "error_count": 0,
            "total_cost": 0,
        }

        # Initialize data storage for this conversation
        data_storage = {}

        # Prepare initial message with instructions
        initial_prompt = self.instructions_prompt.format(question=question)

        initial_message = TextInput(text=initial_prompt)
        self.messages: list[InputItem] = [initial_message]

        agent_logger.info(f"\033[1;34m[USER INSTRUCTIONS]\033[0m {initial_prompt}")

        turn_count = 0
        final_answer = None

        while turn_count < self.max_turns:
            turn_count += 1

            try:
                result, turn_metadata, should_continue = await self._process_turn(
                    turn_count, data_storage
                )

                metadata["turns"].append(turn_metadata)

            except MaxContextWindowExceededError:
                agent_logger.warning(
                    "Max Context Window Exceeded. Removing second earliest message from the stack."
                )
                self.messages.pop(1)
                should_continue = True
            except ModelException as e:
                agent_logger.error(f"\033[1;31m[DO NOT RETRY]\033[0m {e}")
                should_continue = False

            except Exception as e:
                agent_logger.error(f"\033[1;31m[ERROR]\033[0m {e}")
                agent_logger.error(
                    f"\033[1;31m[traceback]\033[0m {traceback.format_exc()}"
                )

                # Explain the error to the agent and give them a chance to recover
                error_message = TextInput(
                    text=f"An error occurred: {e}. Please review what happened and try a different approach."
                )
                self.messages.append(error_message)

                should_continue = True

            if not should_continue:
                final_answer = result
                break

        metadata["end_time"] = datetime.now().isoformat()

        if final_answer:
            metadata["final_answer"] = final_answer

        # Merge turn-level statistics into session-level statistics
        metadata = _merge_statistics(metadata)

        # Save results to file
        os.makedirs("logs/trajectories", exist_ok=True)
        log_path = os.path.join("logs", "trajectories", f"{session_id}.json")
        with open(log_path, "w") as f:
            json.dump(metadata, f, indent=2)

        if final_answer:
            return final_answer, metadata
        else:
            return "Max turns reached without final answer.", metadata
