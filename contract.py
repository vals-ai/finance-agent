from pathlib import Path
from typing import Any, override

from agentic_harness.contract import BaseAgentContract


class FinanceAgentContract(BaseAgentContract):
    @property
    def name(self) -> str:
        return "finance-agent"

    @property
    def install_cmd(self) -> str:
        return "bash setup.sh"

    @property
    def secrets(self) -> dict[str, str]:
        model_keys = [
            "OPENAI_API_KEY",
            "GOOGLE_API_KEY",
            "COHERE_API_KEY",
            "ANTHROPIC_API_KEY",
            "MISTRAL_API_KEY",
            "XAI_API_KEY",
            "AI21LABS_API_KEY",
            "FIREWORKS_API_KEY",
            "DEEPSEEK_KEY",
            "AZURE_API_KEY",
            "DASHSCOPE_API_KEY",
            "PERPLEXITY_KEY",
            "ZAI_API_KEY",
            "KIMI_API_KEY",
            "MERCURY_KEY",
            "MINIMAX_API_KEY",
        ]

        return {
            **{key: "prodBenchmarksInfraApiKeys" for key in model_keys},
            "SEC_EDGAR_API_KEY": "localEvalInfraSecApiKey",
            "TRIVIA_API_KEY": "prodEvalInfraTavilyKey",
        }

    @property
    def final_output(self) -> Path:
        return Path("/workspace/logs/finance_agent")

    @override
    def run_cmd(self, problem_statement_path: str, task_id: str, kwargs: dict[str, Any]) -> str:
        model = self._agent_config.model
        if not model:
            raise ValueError("Model is required. Use --model to specify one.")

        args = [
            "finance-agent",
            f"--model {model}",
            f"--question-file {problem_statement_path}",
        ]

        return " ".join(args)


contract = FinanceAgentContract
