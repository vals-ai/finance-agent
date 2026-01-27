from datetime import datetime
from model_library.base import LLMConfig

TOKEN_KEYS = [
    "in_tokens",
    "out_tokens",
    "reasoning_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "total_input_tokens",
    "total_output_tokens",
]

COST_KEYS = [
    "input",
    "output",
    "reasoning",
    "cache_read",
    "cache_write",
    "total_input",
    "total_output",
    "total",
]


def _merge_statistics(metadata: dict) -> dict:
    """
    Merge turn-level statistics into session-level statistics.

    Args:
        metadata (dict): The metadata with turn-level statistics

    Returns:
        dict: Updated metadata with merged statistics
    """
    # Aggregate statistics from all turns
    for turn in metadata["turns"]:
        metadata["total_cost"] += turn["total_cost"]
        for key in TOKEN_KEYS:
            metadata["total_tokens"][key] += turn["combined_metadata"].get(key, 0) or 0

        for key in TOKEN_KEYS:
            metadata["total_tokens_query"][key] += (
                turn["query_metadata"].get(key, 0) or 0
            )

        if "retrieval_metadata" in turn:
            rm = turn["retrieval_metadata"]
            for key in TOKEN_KEYS:
                metadata["total_tokens_retrieval"][key] += rm.get(key, 0) or 0

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


def create_override_config(**kwargs) -> LLMConfig:
    # Filter kwargs to only include valid LLMConfig fields
    valid_kwargs = {k: v for k, v in kwargs.items() if k in LLMConfig.model_fields}

    # hardcode fix for max output tokens
    if "max_output_tokens" in kwargs and "max_tokens" not in kwargs:
        valid_kwargs["max_tokens"] = kwargs["max_output_tokens"]

    return LLMConfig(**valid_kwargs)
