from datetime import datetime

INSTRUCTIONS_PROMPT = """You are a financial agent. Today is April 07, 2025. You are given a question and you need to answer it using the tools provided.
You may not interract with the user.
When you have the answer, you should respond with 'FINAL ANSWER:' followed by your answer.
At the end of your answer, you should provide your sources in a dictionary with the following format:
{{
    "sources": [
        {{
            "url": "https://example.com",
            "name": "Name of the source"
        }},
        ...
    ]
}}

Question:
{question}
"""


def merge_statistics(metadata: dict) -> dict:
    """
    Merge turn-level statistics into session-level statistics.

    Args:
        metadata (dict): The metadata with turn-level statistics

    Returns:
        dict: Updated metadata with merged statistics
    """
    # Reset aggregate values to recalculate
    metadata["total_tokens"] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    metadata["total_tokens_retrieval"] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    metadata["tool_usage"] = {}
    metadata["tool_calls_count"] = 0
    metadata["api_calls_count"] = len(metadata["turns"])
    metadata["error_count"] = 0

    # Aggregate statistics from all turns
    for turn in metadata["turns"]:
        # Aggregate token usage
        metadata["total_tokens"]["prompt_tokens"] += turn["tokens"]["prompt_tokens"]
        metadata["total_tokens"]["completion_tokens"] += turn["tokens"][
            "completion_tokens"
        ]
        metadata["total_tokens"]["total_tokens"] += turn["tokens"]["total_tokens"]
        metadata["total_tokens_retrieval"]["prompt_tokens"] += turn["tokens_retrieval"][
            "prompt_tokens"
        ]
        metadata["total_tokens_retrieval"]["completion_tokens"] += turn[
            "tokens_retrieval"
        ]["completion_tokens"]
        metadata["total_tokens_retrieval"]["total_tokens"] += turn["tokens_retrieval"][
            "total_tokens"
        ]
        # Count errors
        metadata["error_count"] += len(turn["errors"])

        # Aggregate tool usage
        for tool_call in turn["tool_calls"]:
            tool_name = tool_call["tool_name"]
            if tool_name not in metadata["tool_usage"]:
                metadata["tool_usage"][tool_name] = 0
            metadata["tool_usage"][tool_name] += 1
            metadata["tool_calls_count"] += 1

    # Calculate total duration
    if metadata["start_time"] and metadata["end_time"]:
        start = datetime.fromisoformat(metadata["start_time"])
        end = datetime.fromisoformat(metadata["end_time"])
        metadata["total_duration_seconds"] = (end - start).total_seconds()

    return metadata


# Filter out by pattern because all providers dont throw the same exceptions to OpenAI SDK
def is_token_limit_error(error_msg: str) -> bool:
    token_limit_patterns = [
        "token limit",
        "tokens_exceeded_error",
        "context length",
        "maximum context length",
        "token_limit_exceeded",
        "maximum tokens",
        "too many tokens",
        "prompt is too long",
        "maximum prompt length",
        "maximum number of tokens allowed",
        "input length and `max_tokens` exceed context",
        "error code: 400",
        "error code: 413",
        "string too long",
        "413",
        "request exceeds the maximum size",
        "request_too_large",
        "too many total text bytes",
    ]

    return any(pattern in error_msg.lower() for pattern in token_limit_patterns)
