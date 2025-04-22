import os
import uuid
import json
from datetime import datetime
from abc import ABC
import re

from .tool import Tool
from .llm import GeneralLLM
from .logger import get_logger
from .utils import INSTRUCTIONS_PROMPT, _merge_statistics

agent_logger = get_logger(__name__)


class Agent(ABC):
    def __init__(
        self,
        tools: dict[str, Tool],
        llm: GeneralLLM,
        max_turns: int = 20,
        instructions_prompt: str = INSTRUCTIONS_PROMPT,
    ):
        self.tools = tools
        self.llm = llm
        self.max_turns = max_turns
        self.instructions_prompt = instructions_prompt

    def get_tool_definitions(self) -> list[str]:
        tool_definitions = []
        for name, tool in self.tools.items():
            if hasattr(tool, "get_tool_json"):
                tool_definitions.append(
                    tool.get_tool_json(
                        provider=self.llm.provider,
                        strict=not "grok" in self.llm.model_name
                        and not "deepseek" in self.llm.model_name,
                    )
                )
        return tool_definitions

    async def _process_turn(self, messages, turn_count, data_storage, metadata):
        """
        Process a single turn in the agent's conversation.

        Args:
            messages (list): The conversation history
            turn_count (int): The current turn number
            data_storage (dict): Storage for conversation data
            metadata (dict): Session metadata

        Returns:
            tuple: (final_answer, turn_metadata, should_continue)
        """
        agent_logger.info(f"\033[1;34m[TURN {turn_count}]\033[0m")

        # Initialize turn metadata
        turn_start_time = datetime.now()
        turn_metadata = {
            "turn_id": turn_count,
            "start_time": turn_start_time.isoformat(),
            "end_time": None,
            "duration_seconds": None,
            "tokens": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "tokens_retrieval": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "tool_calls": [],
            "errors": [],
        }

        # Get response from LLM
        response = await self.llm.safe_chat(
            messages=messages, tools=self.get_tool_definitions()
        )

        # Update token usage if available in response
        if hasattr(response, "usage"):
            converted_usage = self.llm.convert_usage(response.usage)
            turn_metadata["tokens"]["prompt_tokens"] = converted_usage["prompt_tokens"]
            turn_metadata["tokens"]["completion_tokens"] = converted_usage[
                "completion_tokens"
            ]
            turn_metadata["tokens"]["total_tokens"] = converted_usage["total_tokens"]

        if self.llm.provider != "anthropic":
            if response is None or response.choices is None:
                agent_logger.error(
                    f"\033[1;31m[LLM STOPPED]\033[0m the agent stopped the conversation before reaching the maximum number of turns or a FINAL ANSWER was found."
                )
                return None, turn_metadata, False
            if (
                response.choices[0].message.content is None
                and response.choices[0].message.tool_calls is None
            ):
                agent_logger.error(
                    f"\033[1;31m[LLM STOPPED]\033[0m the agent stopped the conversation before reaching the maximum number of turns or a FINAL ANSWER was found."
                )
                return None, turn_metadata, False

            if response.choices[0].message.content is not None:
                agent_logger.info(
                    f"\033[1;33m[LLM THINKING]\033[0m {response.choices[0].message.content}"
                )

            if (
                "command-a" in self.llm.model_name
                and response.choices[0].message.content is None
            ):
                response.choices[0].message.content = ""

            messages.append(response.choices[0].message)

        elif self.llm.provider == "anthropic":
            if response.content is None:
                agent_logger.error(
                    f"\033[1;31m[LLM STOPPED]\033[0m the agent stopped the conversation before reaching the maximum number of turns or a FINAL ANSWER was found."
                )
                return None, turn_metadata, False

            if response.content and len(response.content) > 0:
                for block in response.content:
                    if block.type == "text" and block.text:
                        agent_logger.info(
                            f"\033[1;33m[LLM THINKING]\033[0m {block.text}"
                        )

            # Convert Anthropic response to a message format compatible with the conversation
            message = {
                "role": "assistant",
                "content": response.content,
            }
            messages.append(message)

        # Extract tool calls
        tool_calls = self.llm.get_tool_calls(response)

        if tool_calls:
            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                arguments = tool_call["arguments"]
                tool_content = tool_call["tool_content"]

                # Track tool call in turn metadata
                tool_call_metadata = {
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "success": False,
                    "error": None,
                }

                if tool_name not in self.tools:
                    error_msg = f"Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}"

                    # Update error tracking
                    tool_call_metadata["error"] = error_msg
                    turn_metadata["errors"].append(error_msg)

                    # Add error to messages
                    self.llm.append_tool_result(messages, tool_content, error_msg)
                    continue

                # Handle different tool calling patterns
                if tool_name == "retrieve_information":
                    tool_result = await self.tools[tool_name](
                        arguments, data_storage, self.llm
                    )
                    if "usage" in tool_result:
                        tool_token_usage = tool_result["usage"]
                        turn_metadata["tokens"]["prompt_tokens"] += tool_token_usage[
                            "prompt_tokens"
                        ]
                        turn_metadata["tokens"][
                            "completion_tokens"
                        ] += tool_token_usage["completion_tokens"]
                        turn_metadata["tokens"]["total_tokens"] += tool_token_usage[
                            "total_tokens"
                        ]
                        turn_metadata["tokens_retrieval"][
                            "prompt_tokens"
                        ] += tool_token_usage["prompt_tokens"]
                        turn_metadata["tokens_retrieval"][
                            "completion_tokens"
                        ] += tool_token_usage["completion_tokens"]
                        turn_metadata["tokens_retrieval"][
                            "total_tokens"
                        ] += tool_token_usage["total_tokens"]
                elif tool_name == "parse_html_page":
                    tool_result = await self.tools[tool_name](arguments, data_storage)
                else:
                    tool_result = await self.tools[tool_name](arguments)

                if tool_result["success"]:
                    # Add tool result to messages
                    tool_call_metadata["success"] = True
                else:
                    tool_call_metadata["error"] = tool_result["result"]
                    turn_metadata["errors"].append(tool_result["result"])

                self.llm.append_tool_result(
                    messages, tool_content, tool_result["result"]
                )

                # Add tool call metadata to turn
                turn_metadata["tool_calls"].append(tool_call_metadata)

        else:
            # Get text response when there are no tool calls
            response_text = self.llm.parse_response(response)

            # Use regex to check for "FINAL ANSWER:" pattern
            final_answer_pattern = re.compile(r"FINAL ANSWER:", re.IGNORECASE)

            if isinstance(response_text, str) and final_answer_pattern.search(
                response_text
            ):
                # Use regex to extract the content after "FINAL ANSWER:"
                final_answer_match = re.search(
                    r"FINAL ANSWER:(.*?)(?:\{\"sources\"|\Z)",
                    response_text,
                    re.DOTALL,
                )
                sources_match = re.search(
                    r"(\{\"sources\".*\})", response_text, re.DOTALL
                )

                answer_text = (
                    final_answer_match.group(1).strip() if final_answer_match else ""
                )

                # Extract sources if available
                sources_text = sources_match.group(1) if sources_match else ""

                # Combine answer and sources
                final_answer = answer_text
                if sources_text:
                    final_answer = f"{answer_text}\n\n{sources_text}"

                agent_logger.info(f"\033[1;32m[FINAL ANSWER]\033[0m {final_answer}")

                # Finalize turn metadata
                turn_end_time = datetime.now()
                turn_metadata["end_time"] = turn_end_time.isoformat()
                turn_metadata["duration_seconds"] = (
                    turn_end_time - turn_start_time
                ).total_seconds()

                return final_answer, turn_metadata, False
            else:
                agent_logger.info(f"\033[1;33m[LLM THINKING]\033[0m {response_text}")

        # Finalize turn metadata
        turn_end_time = datetime.now()
        turn_metadata["end_time"] = turn_end_time.isoformat()
        turn_metadata["duration_seconds"] = (
            turn_end_time - turn_start_time
        ).total_seconds()

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
            "total_duration_seconds": None,
            "total_tokens": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "total_tokens_retrieval": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "turns": [],
            "tool_usage": {},
            "tool_calls_count": 0,
            "api_calls_count": 0,
            "error_count": 0,
        }

        # Initialize data storage for this conversation
        data_storage = {}

        # Prepare initial message with instructions
        messages = [
            {
                "role": "user",
                "content": self.instructions_prompt.format(question=question),
            }
        ]

        agent_logger.info(
            f"\033[1;34m[USER INSTRUCTIONS]\033[0m {messages[0]['content']}"
        )

        turn_count = 0
        final_answer = None

        while turn_count < self.max_turns:
            turn_count += 1
            # Process the current turn
            result, turn_metadata, should_continue = await self._process_turn(
                messages, turn_count, data_storage, metadata
            )

            # Add turn metadata to session metadata
            metadata["turns"].append(turn_metadata)

            # Check if we should continue or if we have a final answer
            if not should_continue:
                final_answer = result
                break

        # Finalize session metadata
        metadata["end_time"] = datetime.now().isoformat()

        # Add final answer to metadata if available
        if final_answer:
            metadata["final_answer"] = final_answer

        # Merge turn-level statistics into session-level statistics
        metadata = _merge_statistics(metadata)

        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)

        # Save metadata to logs/{session_id}.json
        log_path = os.path.join("logs", f"{session_id}.json")
        with open(log_path, "w") as f:
            json.dump(metadata, f, indent=2)

        if final_answer:
            return final_answer, metadata
        else:
            return "Max turns reached without final answer.", metadata
