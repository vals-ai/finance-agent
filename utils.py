from datetime import datetime

TOKEN_KEYS = [
    "in_tokens",
    "out_tokens",
    "reasoning_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
]


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
        for key in TOKEN_KEYS:
            print(turn["query_metadata"])
            metadata["total_tokens"][key] += turn["query_metadata"].get(key, 0) or 0
            metadata["total_tokens"]["total_tokens"] += (
                turn["query_metadata"].get(key, 0) or 0
            )

        if "retrieval_metadata" in turn:
            rm = turn["retrieval_metadata"]
            for key in TOKEN_KEYS:
                metadata["total_tokens_retrieval"][key] += rm.get(key, 0) or 0
                metadata["total_tokens_retrieval"]["total_tokens"] += (
                    rm.get(key, 0) or 0
                )

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
