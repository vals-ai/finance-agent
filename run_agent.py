import argparse
import asyncio
import json
import os
from datetime import datetime

from agent import agent_logger
from get_agent import get_agent
from tools import tool_logger
from tqdm.asyncio import tqdm


async def run_tests_parallel(
    output_dir,
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
        output_file = os.path.join(output_dir, f"results_test_{timestamp}.json")

        with open(output_file, "w") as f:
            json.dump(formatted_results, f, indent=2)

    return formatted_results


def main():
    parser = argparse.ArgumentParser(
        description="Run the harness for the finance agent benchmark"
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=8192,
        help="Maximum number of output tokens for completion generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for model generation",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    parser.add_argument(
        "--questions", type=str, nargs="+", help="List of questions to process"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="anthropic/claude-3-7-sonnet-20250219",
        help="Model to use to generate completions",
    )
    parser.add_argument(
        "--question-file",
        type=str,
        help="Path to file containing questions (one per line)",
    )
    parser.add_argument(
        "--tools",
        type=str,
        nargs="+",
        default=[
            "google_web_search",
            "retrieve_information",
            "parse_html_page",
            "edgar_search",
        ],
        choices=[
            "google_web_search",
            "retrieve_information",
            "parse_html_page",
            "edgar_search",
        ],
        help="List of tools to make available to the agent",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=50,
        help="Maximum number of turns for the agent to take before stopping",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to save results to.",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=1,
        help="Number of parallel requests to make to the model",
    )
    args = parser.parse_args()

    # Set logging level
    logging_level = args.log_level
    tool_logger.setLevel(logging_level)
    agent_logger.setLevel(logging_level)

    # Get questions from file if provided, otherwise use command line args
    if args.question_file:
        with open(args.question_file, "r") as f:
            questions = [line.strip() for line in f if line.strip()]
    elif args.questions:
        questions = args.questions
    else:
        raise Exception(
            "No questions provided. One of --question-file or --questions must be used."
        )

    parameters = {
        "max_output_tokens": args.max_output_tokens,
        "temperature": args.temperature,
        "max_turns": args.max_turns,
        "tools": args.tools,
    }

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir, exist_ok=True)

    asyncio.run(
        run_tests_parallel(
            output_dir=args.results_dir,
            questions=questions,
            model_name=args.model,
            max_concurrent=args.parallelism,
            save_results=True,
            parameters=parameters,
        )
    )


if __name__ == "__main__":
    main()
