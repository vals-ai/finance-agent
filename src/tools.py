import json
import logging
import os
import re
from typing import Any

import aiohttp
import backoff
from bs4 import BeautifulSoup
from model_library.agent import Tool, ToolOutput
from model_library.base import LLM
from tavily import AsyncTavilyClient

MAX_END_DATE = "2025-04-07"
VALID_TOOLS = ["web_search", "retrieve_information", "parse_html_page", "edgar_search"]


def is_429(exception: Exception) -> bool:
    return isinstance(exception, aiohttp.ClientResponseError) and exception.status == 429 or "429" in str(exception)


def retry_on_429(func):
    @backoff.on_exception(
        backoff.expo,
        aiohttp.ClientResponseError,
        max_tries=100,
        max_value=120,
        base=2,
        factor=3,
        jitter=backoff.full_jitter,
        giveup=lambda e: not is_429(e),
    )
    async def wrapper(*args: object, **kwargs: object):
        return await func(*args, **kwargs)

    return wrapper


class SubmitFinalResult(Tool):
    def __init__(self):
        super().__init__(
            name="submit_final_result",
            description="""
    Submits the final answer to the user. You should include your final answer, as well as any necessary
    reasoning, justification, calculations, and explanation. Finally, you should provide any sources used to answer the question.

    You MUST use this tool to submit your final result. The user will not see your response if you do not use this tool to submit.
    You will not be able to continue working after this tool is called; the conversation will be ended.
    """.strip(),
            parameters={
                "final_result": {
                    "type": "string",
                    "description": "The final result to submit to the agent",
                }
            },
            required=["final_result"],
        )

    async def execute(self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger) -> ToolOutput:
        final_result = args.get("final_result")
        if not final_result:
            return ToolOutput(output="Final result is required", error="Final result is required")
        return ToolOutput(output=final_result, done=True)


class TavilyWebSearch(Tool):
    def __init__(self, tavily_api_key: str | None = None):
        super().__init__(
            name="web_search",
            description="Search the public internet for information. Each result will contain a url, a title, and one excerpt taken directly from the page.",
            parameters={
                "search_query": {
                    "type": "string",
                    "description": "The query to search for",
                },
                "start_date": {
                    "type": "string",
                    "description": "(optional) The start date for the search range for in the format YYYY-MM-DD",
                },
                "end_date": {
                    "type": "string",
                    "description": "(optional) The end date to search range in the format YYYY-MM-DD. If the value is later than 2025-04-07, it will be set to 2025-04-07.",
                },
                "number_of_results": {
                    "type": "integer",
                    "description": "(optional) The number of search results to return.",
                    "maximum": 20,
                    "minimum": 1,
                    "default": 10,
                },
            },
            required=["search_query"],
        )
        if not tavily_api_key:
            tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            raise ValueError("TAVILY_API_KEY is not set")
        self.client = AsyncTavilyClient(api_key=tavily_api_key)

    @retry_on_429
    async def _execute_search(
        self,
        search_query: str,
        start_date: str | None = None,
        end_date: str = MAX_END_DATE,
        number_of_results: int = 10,
    ) -> list[dict[str, Any]]:
        DATE_REGEX = r"^\d{4}-\d{2}-\d{2}$"

        kwargs = {}

        if end_date:
            if not re.match(DATE_REGEX, end_date):
                raise ValueError(f"Invalid end_date format: '{end_date}'. Expected YYYY-MM-DD.")

            if end_date > MAX_END_DATE:
                end_date = MAX_END_DATE

        if start_date:
            if not re.match(DATE_REGEX, start_date):
                raise ValueError(f"Invalid start_date format: '{start_date}'. Expected YYYY-MM-DD.")
            if start_date > MAX_END_DATE:
                start_date = MAX_END_DATE
            if start_date > end_date:
                raise ValueError(
                    f"Parameter start_date '{start_date}' was set to a date that is later than end_date '{end_date}'"
                )

            kwargs["start_date"] = start_date

        response = await self.client.search(
            search_depth="fast",
            end_date=end_date,
            max_results=number_of_results,
            chunks_per_source=1,
            query=search_query,
            **kwargs,
        )

        return response.get("results", [])

    async def execute(self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger) -> ToolOutput:
        results = await self._execute_search(**args)
        return ToolOutput(output=json.dumps(results, default=str))


class EDGARSearch(Tool):
    def __init__(self, sec_api_key: str | None = None):
        super().__init__(
            name="edgar_search",
            description="""
    Search the EDGAR Database through the SEC API.
    You should provide a search query. You can also optionally provide a start date, an end date, a page number, top N results, a list of form types, and/or a list of CIKs.
    The results are returned as a list of dictionaries, each containing the metadata for a filing. It does not contain the full text of the filing.
    """.strip(),
            parameters={
                "search_query": {
                    "type": "string",
                    "description": 'The case-insensitive search-term or phrase to search the contents of fillings and their attachments. This can be a single word, phrase, or combination of words and phrases. Supported search features include wildcards (*), Boolean operators (OR, NOT), and exact phrase matching by enclosing phrases in quotation marks ("exact phrase"). By default, all terms are joined by an implicit AND operator.',
                },
                "form_types": {
                    "type": "array",
                    "description": "(optional) Limits search to specific EDGAR form types (e.g., ['8-K', '10-Q']) list of strings. Default: all form types",
                    "items": {"type": "string"},
                },
                "ciks": {
                    "type": "array",
                    "description": '(optional) Filters results to filings from specified CIKs, type list of strings. Leading zeros are optional but may be included. Example: [ "0001811414", "1318605" ]. Default: all CIKs',
                    "items": {"type": "string"},
                },
                "start_date": {
                    "type": "string",
                    "description": "(optional) Start date for the search range in yyyy-mm-dd format. If the value is a date that is later than 2025-04-07, it will be set to 2025-04-07.",
                    "default": "1900-01-01",
                },
                "end_date": {
                    "type": "string",
                    "description": "(optional) End date for the search range, in the same format as startDate. If the value is a date that is later than 2025-04-07, it will be set to 2025-04-07.",
                    "default": MAX_END_DATE,
                },
                "page": {
                    "type": "string",
                    "description": "(optional) Used for pagination. Each page contains up to 100 matching filings. Increase the page number to retrieve the next set of 100 filings. Example: 3 retrieves the third page. Default: 1",
                    "default": 1,
                },
                "top_n_results": {
                    "type": "integer",
                    "description": "(optional) Return only the first N results out of 100 from the page. If not provided, all 100 results will be returned. E.g. if page is 2, and number_of_results is 10, you will receive results 100 to 110.",
                    "maximum": 100,
                    "default": 100,
                },
            },
            required=["search_query"],
        )
        if sec_api_key is None:
            sec_api_key = os.getenv("SEC_EDGAR_API_KEY")
        if not sec_api_key:
            raise ValueError("SEC_EDGAR_API_KEY is not set")
        self.sec_api_key: str = sec_api_key
        self.sec_api_url: str = "https://api.sec-api.io/full-text-search"

    @retry_on_429
    async def _execute_search(
        self,
        search_query: str,
        start_date: str = "1900-01-01",
        end_date: str = MAX_END_DATE,
        top_n_results: int = 100,
        page: int = 1,
        form_types: list[str] | str | None = None,
        ciks: list[str] | str | None = None,
    ) -> list[str]:

        if form_types is not None and not isinstance(form_types, list):
            raise ValueError(f"The parameter form_types must be a list if provided. Was of type {type(form_types)}")

        if ciks is not None and not isinstance(ciks, list):
            raise ValueError(f"The parameterciks must be an list if provided. Was of type {type(ciks)}")

        date_pattern = r"^\d{4}-\d{2}-\d{2}$"
        if not re.match(date_pattern, start_date):
            raise ValueError(f"start_date '{start_date}' is not in yyyy-mm-dd format")

        if not re.match(date_pattern, end_date):
            raise ValueError(f"end_date '{end_date}' is not in yyyy-mm-dd format")

        if start_date > MAX_END_DATE:
            start_date = MAX_END_DATE

        if end_date > MAX_END_DATE:
            end_date = MAX_END_DATE

        if start_date > end_date:
            raise ValueError(
                f"Parameter start_date '{start_date}' was set to a date that is later than end_date '{end_date}'"
            )

        payload: dict[str, str | int | list[str]] = {
            "query": search_query,
            "startDate": start_date,
            "endDate": end_date,
        }

        if page:
            payload["page"] = page

        if form_types:
            payload["formTypes"] = form_types

        if ciks:
            payload["ciks"] = ciks

        headers = {
            "Content-Type": "application/json",
            "Authorization": self.sec_api_key,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.sec_api_url, json=payload, headers=headers) as response:
                response.raise_for_status()
                result = await response.json()

        results = result.get("filings", [])
        results = results[: int(top_n_results)]

        return results

    async def execute(self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger) -> ToolOutput:
        results = await self._execute_search(**args)
        return ToolOutput(output=json.dumps(results, default=str))


class ParseHtmlPage(Tool):
    def __init__(self):
        super().__init__(
            name="parse_html_page",
            description="""
        This tool is used to parse the contents of an HTML page and save it to the agent's data storage system.

        The tool will retrieve the HTML page from the URL provided, then parse it from HTML to plain text.
        Finally, it will save it to the agent's data storage system under the key provided.

        You can use the retrieve_information tool to later retrieve information about the stored page.
    """.strip(),
            parameters={
                "url": {"type": "string", "description": "The URL of the HTML page to parse"},
                "key": {
                    "type": "string",
                    "description": "The key to use when saving the result in the conversation's data storage.",
                },
            },
            required=["url", "key"],
        )

    @retry_on_429
    async def _parse_html_page(self, url: str) -> str:
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=60),
                    headers={"User-Agent": "ValsAI/antoine@vals.ai"},
                ) as response:
                    response.raise_for_status()
                    html_content = await response.text()
            except Exception as e:
                if len(str(e)) == 0:
                    raise TimeoutError(
                        "Timeout error when parsing HTML page after 60 seconds. The URL might be blocked or the server is taking too long to respond."
                    )
                raise

        soup = BeautifulSoup(html_content, "html.parser")
        for script_or_style in soup(["script", "style"]):
            _ = script_or_style.extract()

        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)

        return text

    async def _save_tool_output(self, output: str, key: str, state: dict[str, Any]) -> str:
        if not output:
            raise ValueError("HTML output was empty")

        tool_result = ""
        if key in state:
            tool_result = (
                "WARNING: The key already exists in the data storage. The new result overwrites the old one.\n"
            )
        tool_result += f"SUCCESS: The result has been saved to the data storage under the key: {key}." + "\n"

        state[key] = output

        keys_list = "\n".join(state.keys())
        tool_result += (
            f"""
        The data_storage currently contains the following keys:
        {keys_list}
        """.strip()
            + "\n"
        )

        return tool_result

    async def execute(self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger) -> ToolOutput:
        url = args.get("url", "")
        if not url:
            raise ValueError("URL is required")
        key = args.get("key", "")
        if not key:
            raise ValueError("Key is required")

        text_output = await self._parse_html_page(url)
        tool_result = await self._save_tool_output(text_output, key, state)
        return ToolOutput(output=tool_result)


class RetrieveInformation(Tool):
    def __init__(self, llm: LLM):
        super().__init__(
            name="retrieve_information",
            description="""
    This tool allows you to retrieve data from previously saved documents from the agent's data storage system, by applying an LLM prompt to the stored document.

    To use the tool, you will need to provide a prompt. This prompt will include both the query to be sent to the LLM,
    as well as the keys of files you have previously saved to the data storage system.

    For example, if you want to analyze data stored under the key "financial_report", your prompt should look like the following:
    "Analyze the following financial report and extract the revenue figures: {{financial_report}}"

    The {{key_name}} will be replaced with the full text of the document stored under that key before the query is sent.

    IMPORTANT: Your prompt MUST include at least one key from the data storage using this exact format: {{key_name}}.
    If you don't use this exact format with double braces, the tool will fail to retrieve the information.

    You can also optionally only pass *a portion* of each document to the LLM, rather than the entire document. This can be used to avoid token limit errors or improve efficiency.
    To do so, use the input_character_ranges parameter to specify which portions of documents to extract.
    For example, if "financial_report" contains "Annual Report 2023" and you specify:  [{"key": "financial_report", "start": 1, "end": 6}], then only "nnual" will be inserted into the prompt (characters 1 through 5, as end is exclusive).
    """.strip(),
            parameters={
                "prompt": {
                    "type": "string",
                    "description": "The prompt that will be passed to the LLM. You MUST include at least one data storage key in the format {{key_name}} - for example: 'Summarize this 10-K filing: {{company_10k}}'. The content stored under each key will replace the {{key_name}} placeholder.",
                },
                "input_character_ranges": {
                    "type": "array",
                    "description": "An optional list of character range specifications for extracting only portions of documents. Each object should have 'key' (the document key), 'start' (start character index, inclusive), and 'end' (end character index, exclusive). By default, the full document is used if this parameter is not provided or if a key is not included in the list.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "key": {
                                "type": "string",
                                "description": "The document key from data storage",
                            },
                            "start": {
                                "type": "integer",
                                "description": "The starting character index (inclusive)",
                            },
                            "end": {
                                "type": "integer",
                                "description": "The ending character index (exclusive)",
                            },
                        },
                        "required": ["key", "start", "end"],
                    },
                },
            },
            required=["prompt"],
        )
        self._llm = llm

    def _validate_inputs(self, prompt: str, input_character_ranges: list, state: dict[str, Any]) -> dict[str, tuple[int, int]]:
        """Validate prompt placeholders, character ranges, and data storage keys. Returns the parsed ranges dict."""
        if not re.search(r"{{[^{}]+}}", prompt):
            raise ValueError(
                "ERROR: Your prompt must include at least one key from data storage in the format {{key_name}}. Please try again with the correct format. You can add documents to the data storage with parse_html_page."
            )

        ranges_dict = {}
        for range_spec in input_character_ranges:
            if not isinstance(range_spec, dict):
                raise ValueError(
                    "ERROR: Each item in input_character_ranges must be an object with 'key', 'start', and 'end' fields."
                )
            if "key" not in range_spec or "start" not in range_spec or "end" not in range_spec:
                raise ValueError("ERROR: Each range specification must have 'key', 'start', and 'end' fields.")
            ranges_dict[range_spec["key"]] = (range_spec["start"], range_spec["end"])

        keys = re.findall(r"{{([^{}]+)}}", prompt)
        keys_set = set(keys)

        for range_key in ranges_dict.keys():
            if range_key not in keys_set:
                raise ValueError(
                    f"ERROR: The key '{range_key}' is specified in input_character_ranges but is not referenced in the prompt. "
                    f"Keys in prompt: {', '.join(keys_set) if keys_set else '(none)'}"
                )

        for key in keys:
            if key not in state:
                raise KeyError(
                    f"ERROR: The key '{key}' was not found in the data storage. Available keys are: {', '.join(state.keys())}. Use the retrieve_information tool to add keys to the data storage."
                )

        return ranges_dict

    def _format_prompt(self, prompt: str, ranges_dict: dict[str, tuple[int, int]], state: dict[str, Any]) -> str:
        """Substitute data storage content into prompt placeholders, applying character ranges."""
        keys = re.findall(r"{{([^{}]+)}}", prompt)
        formatted_data = {}

        for key in keys:
            doc_content = state[key]
            if key in ranges_dict:
                start_idx, end_idx = ranges_dict[key]
                formatted_data[key] = doc_content[start_idx:end_idx]
            else:
                formatted_data[key] = doc_content

        formatted_prompt = re.sub(r"{{([^{}]+)}}", r"{\1}", prompt)

        try:
            return formatted_prompt.format(**formatted_data)
        except KeyError as e:
            raise KeyError(
                f"ERROR: The key {str(e)} was not found in the data storage. Available keys are: {', '.join(state.keys())}"
            )

    async def execute(self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger) -> ToolOutput:
        prompt: str = args["prompt"]
        input_character_ranges = args.get("input_character_ranges", [])
        if input_character_ranges is None:
            input_character_ranges = []

        ranges_dict = self._validate_inputs(prompt, input_character_ranges, state)
        prompt = self._format_prompt(prompt, ranges_dict, state)

        response = await self._llm.query(prompt)

        return ToolOutput(
            output=response.output_text_str,
            metadata=response.metadata,
        )
