# ValsAI Finance Agent Benchmark Codebase

Finance Agent is a tool for financial research and analysis that leverages large language models and specialized financial tools to answer complex queries about companies, financial statements, and SEC filings.

This repo contains the codebase to run the agent that was used to create the benchmark [Finance Agent](https://www.vals.ai/benchmarks/finance_agent). It makes it easy to test the harness with any question or model of your choices.

## Overview

This agent connects to various data sources including:

- SEC EDGAR database
- Web search (via Tavily)
- HTML page parsing capabilities
- Information retrieval and analysis

It uses a configurable LLM backend (OpenAI, Anthropic, Google, etc.) to orchestrate these tools and generate comprehensive financial analysis.
For all models except Anthropic's, we use OpenAI SDK for the agent.

## Installation

Run the following command to install the enviromnent:

```
pip install -r requirements.txt
```

We recommend creating and using a Conda environment for this. For detailed instructions on managing Conda environments, see the [official Conda documentation](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html).

## Environment Setup

Create a `.env` file with the necessary API keys:

```
# LLM API Keys
# Note: It's only necessary to set the API keys for the models you plan on using
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
MISTRAL_API_KEY=your_mistral_key
TOGETHER_API_KEY=your_together_api_key
FIREWORKS_API_KEY=your_fireworks_api_key
XAI_API_KEY=your_grok_api_key
COHERE_API_KEY=cohere_api_key

# Tool API Keys
TAVILY_API_KEY=your_tavily_key
SEC_EDGAR_API_KEY=your_sec_api_key
```

You can create a Tavily API key [here](https://tavily.com/), and an SEC API key [here](https://sec-api.io/).

## Running the Agent

To experiment with the agent, you can run the following command:

```bash
python run_agent.py --questions "What was Apple's revenue in 2023?"
```

You can specify multiple questions at once:

```bash
python run_agent.py --questions "What was Apple's revenue in 2023?" "What was NFLX's revenue in 2024?"
```

To specify a specific model, use the `--model` flag:

```bash
python run_agent.py --questions "What was Apple's revenue in 2023?" --model openai/gpt-4o
```

You can also specify a list of questions in a text file, one question per line, with the following command:

```bash
python run_agent.py --question-file data/public.txt
```

For a full list of parameters, please run:

```bash
python run_agent.py --help
```

The default configuration is the one we used to run the benchmark.

## Available Tools

- `web_search`: Search the web for information
- `edgar_search`: Search the SEC's EDGAR database for filings
- `parse_html_page`: Parse and extract content from web pages
- `retrieve_information`: Access stored information from previous steps

## Available Models

The harness use's the Vals [Model Library](https://github.com/vals-ai/model-library), an open-source repository that allows you to call models in a unified manner.

Generally, any model that is in the Vals Model Library (and that supports tool calling) can be used by the agent. Here are a few commonly used models:

```
openai/gpt-5.1-2025-11-13
anthropic/claude-sonnet-4-5-20250929-thinking
google/gemini-2.5-pro
grok/grok-4-fast-reasoning
cohere/command-a-03-2025
zai/glm-4.6
kimi/kimi-k2-thinking
fireworks/glm-4p6
```

Full documentation is available in the model library repo. You can find the full list of models [here](https://github.com/vals-ai/model-library/blob/main/model_library/config/all_models.json).

## Logs and Output

The agent writes detailed logs to the `logs` directory, including:

- Session-specific logs with timestamps
- Tool usage statistics
- Token usage
- Error tracking
