# ValsAI Finance Agent Benchmark Codebase

Finance Agent is a tool for financial research and analysis that leverages large language models and specialized financial tools to answer complex queries about companies, financial statements, and SEC filings.

This repo contains the codebase to run the agent that was used to create the benchmark [Finance Agent](https://www.vals.ai/benchmarks/finance_agent). It makes it easy to test the harness with any question or model of your choices.

## Overview

This agent connects to various data sources including:
- SEC EDGAR database
- Google web search
- HTML page parsing capabilities
- Information retrieval and analysis

It uses a configurable LLM backend (OpenAI, Anthropic, Google, etc.) to orchestrate these tools and generate comprehensive financial analysis.
For all models except Anthropic's, we use OpenAI SDK for the agent.

## Installation

Run the following command to install the enviromnent:
```
pip install -r requirements.txt
```

## Configuration

The agent is configured using the `run_config.yaml` file. Here's an explanation of the key parameters:

```yaml
# The LLM provider and model to use
model_id: google/gemini-2.5-pro-exp-03-25

# The benchmark suite ID to run (if using vals benchmarking)
suite_id: edgar_research_final_review

# Agent parameters
agent_parameters:
  # Maximum number of interaction turns before stopping
  max_turns: 50

  # Tools available to the agent
  available_tools:
    - google_web_search
    - retrieve_information
    - parse_html_page
    - edgar_search

# Parameters for benchmark runs
run_parameters:
  parallelism: 10
```

### Configuration Options

- **model_id**: Specifies the provider and model name in the format `provider/model_name`
  - Supported providers: openai, anthropic, google, mistralai, together, fireworks, grok
  - Examples: openai/gpt-4, anthropic/claude-3-7-sonnet-20250219, google/gemini-2.5-pro

- **agent_parameters**:
  - **max_turns**: Maximum number of tool-use iterations before the agent terminates
  - **available_tools**: List of tools the agent can use (must be defined in tool.py)

## Environment Setup

Create a `.env` file with the necessary API keys:

```
# LLM API Keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
MISTRAL_API_KEY=your_mistral_key
# ...other provider keys as needed

# Tool API Keys
SERPAPI_API_KEY=your_serpapi_key
SEC_API_KEY=your_sec_api_key
```

## Running the Agent

To expeiment with the agent, simply configure the question(s), model and parameters you want to test in `run_agent.py` and then run the following command:

```
python -m edgar-finance-agent.run_agent
```

For more details parameters, please look at `get_agent.py`. The default configuration is the one we used to run the benchmark.

## Available Tools

- `google_web_search`: Search the web for information
- `edgar_search`: Search the SEC's EDGAR database for filings
- `parse_html_page`: Parse and extract content from web pages
- `retrieve_information`: Access stored information from previous steps

## Logs and Output

The agent writes detailed logs to the `logs` directory, including:
- Session-specific logs with timestamps
- Tool usage statistics
- Token usage
- Error tracking