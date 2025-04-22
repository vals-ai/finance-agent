import json
import os
import backoff

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from abc import ABC, abstractmethod
from openai.types.chat import ChatCompletion

from .utils import is_token_limit_error

provider_args = {
    "openai": {"api_key": os.getenv("OPENAI_API_KEY")},
    "google": {
        "api_key": os.getenv("GOOGLE_API_KEY"),
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
    "anthropic": {
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
        "base_url": "https://api.anthropic.com/v1/",
    },
    "together": {
        "api_key": os.getenv("TOGETHER_API_KEY"),
        "base_url": "https://api.together.xyz/v1/",
    },
    "fireworks": {
        "api_key": os.getenv("FIREWORKS_API_KEY"),
        "base_url": "https://api.fireworks.ai/v1/",
    },
    "mistralai": {
        "api_key": os.getenv("MISTRAL_API_KEY"),
        "base_url": "https://api.mistral.ai/v1/",
    },
    "grok": {
        "api_key": os.getenv("GROK_API_KEY"),
        "base_url": "https://api.x.ai/v1",
    },
    "cohere": {
        "api_key": os.getenv("COHERE_API_KEY"),
        "base_url": "https://api.cohere.ai/compatibility/v1/",
    },
}


class LLM(ABC):
    def __init__(self, provider: str, model_name: str):
        self.provider = provider
        self.model_name = model_name
        params = self.get_provider_args()
        if self.provider == "fireworks":
            self.model_name = "accounts/fireworks/models/" + self.model_name
        self.client = AsyncOpenAI(
            api_key=params["api_key"], base_url=params.get("base_url")
        )
        # self.provider = provider if provider == "anthropic" else "openai"

    def get_provider_args(self) -> dict[str, any]:
        params = provider_args[self.provider]
        return params

    def chat(self, conversation: list[dict[str, any]]) -> str:
        pass

    @abstractmethod
    def parse_response(self, response: dict[str, any]) -> str:
        pass

    @abstractmethod
    def append_tool_result(
        self,
        messages: list[dict[str, any]],
        tool_content: any,
        tool_result: str,
    ) -> list[dict[str, any]]:
        pass

    @abstractmethod
    def get_tool_calls(self, response: dict[str, any]) -> list[dict[str, any]]:
        pass

    def convert_usage(self, usage: dict[str, any]) -> dict[str, any]:
        pass


class GeneralLLM(LLM):
    def __init__(
        self,
        provider: str,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ):
        super().__init__(provider=provider, model_name=model_name)
        self.temperature = temperature
        self.max_tokens = max_tokens
        if provider == "anthropic":
            params = self.get_provider_args()
            self.anthropic_client = AsyncAnthropic(api_key=params["api_key"])

    async def safe_chat(
        self,
        messages: list[dict[str, any]],
        tools: list[dict[str, any]] = [],
        ignore_token_error: bool = False,
    ) -> ChatCompletion:
        while True:
            try:
                # Use the retryable chat function with original messages
                return await self._retryable_chat(messages, tools)
            except Exception as e:
                error_str = str(e).lower()

                # Handle token limit errors specially
                if is_token_limit_error(error_str):
                    if ignore_token_error:
                        raise Exception(f"Token limit error: {str(e)}")
                    if len(messages) > 2:
                        print(
                            f"Too long, removing oldest message pair. {len(messages)}"
                        )
                        messages.pop(1)
                        continue
                    else:
                        raise Exception(
                            f"Cannot reduce message context any further: {str(e)}"
                        )

                raise

    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=None,
        max_value=150,
        max_time=1200,  # Maximum backoff time in seconds
        jitter=backoff.full_jitter,  # Add jitter to avoid thundering herd
        giveup=lambda e: is_token_limit_error(
            str(e).lower()
        ),  # Use the shared function
    )
    async def _retryable_chat(
        self, messages: list[dict[str, any]], tools: list[dict[str, any]]
    ) -> ChatCompletion:
        try:
            return await self.chat(messages, tools)
        except Exception as e:
            print(f"Error: {e}")
            raise

    async def chat(
        self, messages: list[dict[str, any]], tools: list[dict[str, any]] = []
    ) -> ChatCompletion:
        if self.provider == "anthropic":
            if self.model_name == "claude-3-7-sonnet-20250219-thinking":
                model_name = "claude-3-7-sonnet-20250219"
                if len(tools) > 0:
                    return await self.anthropic_client.messages.create(
                        model=model_name,
                        messages=messages,
                        temperature=1,
                        tools=tools,
                        thinking={"type": "enabled", "budget_tokens": 13384},
                        max_tokens=16384,
                    )
                else:
                    return await self.anthropic_client.messages.create(
                        model=model_name,
                        messages=messages,
                        temperature=1,
                        max_tokens=16384,
                        thinking={"type": "enabled", "budget_tokens": 13384},
                    )
            else:
                if len(tools) > 0:
                    return await self.anthropic_client.messages.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=self.temperature,
                        tools=tools,
                        max_tokens=8192,
                    )
                else:
                    return await self.anthropic_client.messages.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=8192,
                    )

        if self.model_name in [
            "o3-mini-2025-01-31",
            "o1-2024-12-17",
            "o3-2025-04-16",
            "o4-mini-2025-04-16",
        ]:
            if len(tools) > 0:
                return await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=tools,
                )
            else:
                return await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                )

        if self.model_name == "grok-3-mini-fast-beta-high-reasoning":
            if len(tools) > 0:
                return await self.client.chat.completions.create(
                    model="grok-3-mini-fast-beta",
                    messages=messages,
                    temperature=self.temperature,
                    tools=tools,
                    max_tokens=self.max_tokens,
                    reasoning_effort="high",
                )
            else:
                return await self.client.chat.completions.create(
                    model="grok-3-mini-fast-beta",
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    reasoning_effort="high",
                )

        if self.model_name == "grok-3-mini-fast-beta-low-reasoning":
            if len(tools) > 0:
                return await self.client.chat.completions.create(
                    model="grok-3-mini-fast-beta",
                    messages=messages,
                    temperature=self.temperature,
                    tools=tools,
                    max_tokens=self.max_tokens,
                    reasoning_effort="low",
                )
            else:
                return await self.client.chat.completions.create(
                    model="grok-3-mini-fast-beta",
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    reasoning_effort="low",
                )

        if len(tools) > 0:
            return await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                tools=tools,
                max_tokens=self.max_tokens,
            )
        else:
            return await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

    def get_tool_calls(self, response: dict[str, any]) -> list[dict[str, any]]:
        tools = []

        # Handle Anthropic response format
        if self.provider == "anthropic":
            for content in response.content:
                if content.type == "tool_use":
                    tools.append(
                        {
                            "name": content.name,
                            "arguments": content.input,
                            "tool_content": content,
                        }
                    )
            return tools

        # Handle OpenAI response format
        for choice in response.choices:
            if choice.message.tool_calls:
                for tool_call in choice.message.tool_calls:
                    tools.append(
                        {
                            "name": tool_call.function.name,
                            "arguments": json.loads(tool_call.function.arguments),
                            "tool_content": tool_call,
                        }
                    )
        return tools

    def parse_response(self, response: dict[str, any]) -> str:
        # Handle Anthropic response format
        if self.provider == "anthropic":
            for content in response.content:
                if content.type == "text":
                    return content.text
            return ""  # Return empty string if no text content found

        # Handle OpenAI response format
        return response.choices[0].message.content

    def append_tool_result(
        self,
        messages: list[dict[str, any]],
        tool_content: any,
        tool_result: str,
    ) -> list[dict[str, any]]:
        # Handle Anthropic response format
        if self.provider == "anthropic":
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_content.id,
                            "content": tool_result,
                        }
                    ],
                }
            )
        # Handle OpenAI response format
        else:
            messages.append(
                {
                    "role": "tool",
                    "content": tool_result if tool_result is not None else "",
                    "tool_call_id": tool_content.id,
                }
            )
        return messages

    def convert_usage(self, usage: dict[str, any]) -> dict[str, any]:
        if self.provider == "anthropic":
            return {
                "prompt_tokens": usage.input_tokens,
                "completion_tokens": usage.output_tokens,
                "total_tokens": usage.input_tokens + usage.output_tokens,
            }
        else:
            return {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }
