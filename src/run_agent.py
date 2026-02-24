import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from get_agent import Parameters, get_agent
from model_library.agent import AgentResult
from model_library.base import LLMConfig
from model_library.base.input import TextInput
from model_library.utils import create_file_logger, create_run_dir
from prompt import INSTRUCTIONS_PROMPT
from tools import VALID_TOOLS
from tqdm.asyncio import tqdm


async def run_tests_parallel(
    questions: list[str],
    max_concurrent: int,
    parameters: Parameters,
) -> list[dict[str, Any]]:
    """Run multiple questions in parallel using the agent"""
    run_dir = create_run_dir("finance_agent", parameters.model_name)

    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_question(question: str, question_index: int):
        async with semaphore:
            log_file = run_dir / f"q{question_index:03d}.log"
            with create_file_logger(f"agent.q{question_index:03d}", log_file) as logger:
                agent = get_agent(parameters, logger_name=logger.name)
                prompt = INSTRUCTIONS_PROMPT.format(question=question)
                result = await agent.run([TextInput(text=prompt)])
                return result

    tasks = [process_question(question, i + 1) for i, question in enumerate(questions)]

    results: list[AgentResult] = await tqdm.gather(*tasks, desc="Processing questions")

    formatted_results = []
    for question, result in zip(questions, results):
        if isinstance(result, Exception):
            formatted_results.append({"question": question, "success": False, "error": str(result)})
        else:
            formatted_results.append({"question": question, "success": result.success, "result": result.model_dump()})

    results_file = run_dir / "results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(formatted_results, f, indent=2)

    return formatted_results


async def main():
    parser = argparse.ArgumentParser(description="Run the harness for the finance agent benchmark")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32000,
        help="Maximum number of tokens for completion generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for model generation",
    )
    parser.add_argument("--questions", type=str, nargs="+", help="List of questions to process")
    parser.add_argument(
        "--model",
        type=str,
        default="anthropic/claude-sonnet-4-5-20250929",
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
        default=VALID_TOOLS,
        choices=VALID_TOOLS,
        help="List of tools to make available to the agent",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=50,
        help="Maximum number of turns for the agent to take before stopping",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=1,
        help="Number of parallel requests to make to the model",
    )
    args = parser.parse_args()

    ENV_FILE = Path(".env")
    load_dotenv(override=True, dotenv_path=ENV_FILE)

    if args.question_file:
        with open(args.question_file) as f:
            questions = [line.strip() for line in f if line.strip()]
    elif args.questions:
        questions = args.questions
    else:
        raise Exception("No questions provided. One of --question-file or --questions must be used.")

    parameters = Parameters(
        model_name=args.model,
        max_turns=args.max_turns,
        tools=args.tools,
        llm_config=LLMConfig(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        ),
    )

    await run_tests_parallel(
        questions=questions,
        max_concurrent=args.parallelism,
        parameters=parameters,
    )


if __name__ == "__main__":
    asyncio.run(main())
