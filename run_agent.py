import os
import sys
import json
import asyncio
from datetime import datetime
from tqdm.asyncio import tqdm

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Now relative imports will work
from get_agent import get_agent
from tool import tool_logger
from agent import agent_logger

# Output directory for results
OUTPUT_DIR = f"{parent_dir}/results_tests"
os.makedirs(OUTPUT_DIR, exist_ok=True)


async def run_tests_parallel(
    questions=[],
    model_name="anthropic/claude-3-7-sonnet-20250219",
    max_concurrent=5,
    save_results=False,
    parameters={},
):
    """Run multiple questions in parallel using the custom model"""
    agent = await get_agent(model_name, parameters)

    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_question(question):
        async with semaphore:
            return await agent.run(question)

    tasks = [process_question(question) for question in questions]

    results = await tqdm.gather(*tasks, desc="Processing questions")

    formatted_results = []
    for i, (question, result) in enumerate(zip(questions, results)):
        if isinstance(result, Exception):
            formatted_results.append(
                {"question": question, "success": False, "error": str(result)}
            )
        else:
            formatted_results.append(
                {"question": question, "success": True, "result": result}
            )

    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(OUTPUT_DIR, f"results_test_{timestamp}.json")

        with open(output_file, "w") as f:
            json.dump(formatted_results, f, indent=2)

    return formatted_results


if __name__ == "__main__":
    # Test questions
    test_questions = [
        "Due to its business combinations, what is RTX Corp's (NYSE: RTX) projected future contractual obligation consumption for 2025 - 2029. Provide the amount for each year.",
        "What is the Total Number of Common Stock Shares Repurchased by Netflix (NASDAQ: NFLX) in Q4 2024?",
    ]

    # Try different models
    model_name = "openai/gpt-4o-2024-08-06"
    model_name = "anthropic/claude-3-7-sonnet-20250219"
    model_name = "grok/grok-3-beta"

    # Set parameters
    parameters = {"max_output_tokens": 8192, "temperature": 0.0}

    # Leave logging level to INFO to see the agent's thought process.
    # Set logging level to CRITICAL to suppress all logs.
    logging_level = "INFO"
    tool_logger.setLevel(logging_level)
    agent_logger.setLevel(logging_level)

    asyncio.run(
        run_tests_parallel(
            test_questions,
            model_name,
            max_concurrent=1,
            save_results=True,
            parameters=parameters,
        )
    )
