from datetime import datetime
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from .agent import Metadata


def _merge_statistics(metadata: "Metadata") -> "Metadata":
    """
    Merge turn-level statistics into session-level statistics.

    Args:
        metadata (dict): The metadata with turn-level statistics

    Returns:
        dict: Updated metadata with merged statistics
    """
    # Aggregate statistics from all turns
    for turn in metadata.turns:
        metadata.total_cost += turn.total_cost

        metadata.total_tokens += turn.combined_metadata

        metadata.total_tokens_query += turn.query_metadata

        metadata.total_tokens_retrieval += turn.retrieval_metadata

        metadata.error_count += len(turn.errors)

        # Aggregate tool usage
        for tool_call in turn.tool_calls:
            tool_name = tool_call["tool_name"]
            if tool_name not in metadata.tool_usage:
                metadata.tool_usage[tool_name] = 0
            metadata.tool_usage[tool_name] += 1
            metadata.tool_calls_count += 1

    # Calculate total duration
    if metadata.start_time and metadata.end_time:
        start = datetime.fromisoformat(metadata.start_time)
        end = datetime.fromisoformat(metadata.end_time)
        metadata.total_duration_seconds = (end - start).total_seconds()

    return metadata
