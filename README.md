# Running Finance Agent Benchmark

Our Finance Agent benchmark evaluates LLMs on their ability to use tools to research and answer complex financial questions about companies, financial statements, and SEC filings.

The agent has access to the following tools:

- `web_search`: Search the web for information (via Tavily)
- `edgar_search`: Search the SEC's EDGAR database for filings
- `parse_html_page`: Parse and extract content from web pages
- `retrieve_information`: Access stored information from previous steps

For more details on the benchmark, please refer to our [public website](https://www.vals.ai/benchmarks/finance_agent).

## Set up

### Dependencies

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management. Then run:

```
make install
source .venv/bin/activate
```

### Platform

Make an account on [platform.vals.ai](https://www.platform.vals.ai/auth) with your company email address. Go to the admin page and create a new API key for yourself.

### Environment Variables

Create a `.env` file in the root of the project and add the following:

```
VALS_API_KEY=<api_key>

# LLM API Keys (only set the ones you plan on using)
OPENAI_API_KEY=<openai_api_key>
ANTHROPIC_API_KEY=<anthropic_api_key>
GOOGLE_API_KEY=<google_api_key>
ETC_API_KEY=<etc_api_key>

# Tool API Keys
TAVILY_API_KEY=<tavily_api_key>
SEC_EDGAR_API_KEY=<sec_api_key>
```

You can create a Tavily API key [here](https://tavily.com/), and an SEC API key [here](https://sec-api.io/).

The `.env` takes precedence over set environment variables.

Finally, you should add the "Test Suite IDs" to suites.json. These should have generally been provided to you via email, but you can also find them in the platform, by navigating to the "Test Suites" page, clicking the relevant test suite, and looking on the right sidebar under "Test Suite ID".

## Running the benchmark

For a list of command line options, run `python src/run_agent.py --help`

To run, for example, a single question on openai/gpt-5.2-2025-12-11:

```
python src/run_agent.py --questions "What was Apple's revenue in 2023?" --model openai/gpt-5.2-2025-12-11
```

You can specify multiple questions at once:

```
python src/run_agent.py --questions "What was Apple's revenue in 2023?" "What was NFLX's revenue in 2024?"
```

You can also specify a list of questions in a text file, one question per line:

```
python src/run_agent.py --question-file data/public.txt
```

The default configuration is the one we used to run the benchmark.

### List of Models

A list of available models can be found at our [model library](https://github.com/vals-ai/model-library/blob/main/model_library/config/all_models.json), and also by running `make browse-models` in the model library repository.

To run your own harness or model, just modify the `get_custom_model` function as needed. To see the full documentation on how the SDK works, visit [our docs](https://docs.vals.ai/sdk/running_suites).

## Logs

The agent writes detailed logs to the `logs/` directory. Each run creates a timestamped directory with per-question log files containing tool usage, token counts, and error tracking.
