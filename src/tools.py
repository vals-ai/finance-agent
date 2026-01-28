import json
import os
import re
import traceback
from abc import ABC, abstractmethod
from typing import Any, override

import aiohttp
import backoff
from bs4 import BeautifulSoup
from model_library.base import LLM, ToolBody, ToolDefinition
from pydantic import computed_field

from logger import VERBOSE, get_logger

tool_logger = get_logger(__name__)

MAX_END_DATE = "2025-04-07"


def is_429(exception: Exception) -> bool:
    is429 = (
        isinstance(exception, aiohttp.ClientResponseError)
        and exception.status == 429
        or "429" in str(exception)
    )
    if is429:
        tool_logger.error(f"429 error: {exception}")
    return is429


# Define a reusable backoff decorator for 429 errors. Mainly used for the SEC and Google Search APIs.
def retry_on_429(func):
    @backoff.on_exception(
        backoff.expo,
        aiohttp.ClientResponseError,
        max_tries=8,
        base=2,
        factor=3,
        jitter=backoff.full_jitter,
        giveup=lambda e: not is_429(e),
    )
    async def wrapper(*args: object, **kwargs: object):
        return await func(*args, **kwargs)

    return wrapper


class Tool(ABC):
    """
    Abstract base class for tools.
    """

    name: str
    description: str
    input_arguments: dict[str, Any]
    required_arguments: list[str]

    @computed_field
    @property
    def tool_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            body=ToolBody(
                name=self.name,
                description=self.description,
                properties=self.input_arguments,
                required=self.required_arguments,
            ),
        )

    def __init__(
        self,
        *args: object,
        **kwargs: object,
    ):
        super().__init__()

    @abstractmethod
    async def call_tool(
        self, arguments: dict[str, Any], data_storage: dict[str, Any], llm: LLM
    ) -> dict[str, Any]: ...

    async def __call__(
        self,
        arguments: dict[str, Any],
        data_storage: dict[str, Any],
        llm: LLM,
    ) -> dict[str, Any]:
        """
        Wrapper function to call the subclass' call_tool method
        """
        tool_logger.info(
            f"\033[1;33m[TOOL: {self.name.upper()}]\033[0m Calling with arguments: {arguments}"
        )

        try:
            tool_result = await self.call_tool(arguments, data_storage, llm)
            tool_logger.info(
                f"\033[1;32m[TOOL: {self.name.upper()}]\033[0m Returned: {tool_result}"
            )
            if self.name == "retrieve_information":
                return {
                    "success": True,
                    "result": tool_result["retrieval"],
                    "usage": tool_result["usage"],
                }
            return {"success": True, "result": json.dumps(tool_result)}
        except Exception as e:
            error_msg = str(e)
            if VERBOSE:
                error_msg += f"\nTraceback: {traceback.format_exc()}"
                tool_logger.warning(
                    f"\033[1;31m[TOOL: {self.name.upper()}]\033[0m Error: {e}\nTraceback: {traceback.format_exc()}"
                )
            else:
                tool_logger.warning(
                    f"\033[1;31m[TOOL: {self.name.upper()}]\033[0m Error: {e}"
                )
            return {"success": False, "result": error_msg}


class SubmitFinalResult(Tool):
    name: str = "submit_final_result"
    description: str = """
    Submit the final result to the agent.
    This should only be called once, at the end of the conversation. Calling it will end the conversation.
    You should provide the final result as a string.
    """.strip()
    input_arguments: dict[str, Any] = {
        "final_result": {
            "type": "string",
            "description": "The final result to submit to the agent",
        }
    }
    required_arguments: list[str] = ["final_result"]

    def __init__(self, *args, **kwargs):
        super().__init__()

    async def call_tool(
        self, arguments: dict[str, Any], *args, **kwargs
    ) -> dict[str, Any]:
        final_result = arguments.get("final_result")
        if not final_result:
            return {"success": False, "result": "Final result is required"}

        return {"success": True, "result": final_result}


class GoogleWebSearch(Tool):
    name: str = "google_web_search"
    description: str = "Search the web for information"
    input_arguments: dict[str, Any] = {
        "search_query": {
            "type": "string",
            "description": "The query to search for",
        }
    }
    required_arguments: list[str] = ["search_query"]

    def __init__(
        self,
        top_n_results: int = 10,
        serpapi_api_key: str | None = None,
    ):
        super().__init__(
            self.name,
            self.description,
            self.input_arguments,
            self.required_arguments,
        )
        self.top_n_results: int = top_n_results
        if not serpapi_api_key:
            serpapi_api_key = os.getenv("SERPAPI_API_KEY")
        if not serpapi_api_key:
            raise ValueError("SERPAPI_API_KEY is not set")

        self.serpapi_api_key: str = serpapi_api_key

    @retry_on_429
    async def _execute_search(self, search_query: str) -> list[str]:
        """
        Search the web for information using Google Search.

        Args:
            search_query (str): The query to search for

        Returns:
            list[str]: A list of results from Google Search
        """
        if not self.serpapi_api_key:
            raise ValueError("SERPAPI_API_KEY is not set")

        # Google expect MM/DD/YYYY format
        max_date_parts = MAX_END_DATE.split("-")
        google_date_format = (
            f"{max_date_parts[1]}/{max_date_parts[2]}/{max_date_parts[0]}"
        )

        params = {
            "api_key": self.serpapi_api_key,
            "engine": "google",
            "q": search_query,
            "num": self.top_n_results,
            "tbs": f"cdr:1,cd_max:{google_date_format}",
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://serpapi.com/search.json", params=params
            ) as response:
                response.raise_for_status()  # This will raise ClientResponseError
                results = await response.json()

        return results.get("organic_results", [])

    @override
    async def call_tool(
        self, arguments: dict[str, Any], data_storage: dict[str, Any], llm: LLM
    ) -> dict[str, Any]:
        results = await self._execute_search(**arguments)
        return results


class EDGARSearch(Tool):
    name: str = "edgar_search"
    description: str = """
    Search the EDGAR Database through the SEC API.
    You should provide a query, a start date, an end date, a page number, and a top N results. Optionally provide a list of form types, and a list of CIKs.
    The results are returned as a list of dictionaries, each containing the metadata for a filing. It does not contain the full text of the filing.
    """.strip()
    input_arguments: dict[str, Any] = {
        "query": {
            "type": "string",
            "description": 'The case-insensitive search-term or phrase to search the contents of fillings and their attachments. This can be a single word, phrase, or combination of words and phrases. Supported search features include wildcards (*), Boolean operators (OR, NOT), and exact phrase matching by enclosing phrases in quotation marks ("exact phrase"). By default, all terms are joined by an implicit AND operator.',
        },
        "form_types": {
            "type": "array",
            "description": "Limits search to specific EDGAR form types (e.g., ['8-K', '10-Q']) list of strings. Default: all form types",
            "items": {"type": "string"},
        },
        "ciks": {
            "type": "array",
            "description": 'Filters results to filings from specified CIKs, type list of strings. Leading zeros are optional but may be included. Example: [ "0001811414", "1318605" ]. Default: all CIKs',
            "items": {"type": "string"},
        },
        "start_date": {
            "type": "string",
            "description": "Start date for the search range in yyyy-mm-dd format. Used in combination with endDate to define the date range. Example: '2024-01-01'. Default is 30 days ago",
        },
        "end_date": {
            "type": "string",
            "description": "End date for the search range, in the same format as startDate. Default is today",
        },
        "page": {
            "type": "string",
            "description": "Used for pagination. Each request returns 100 matching filings. Increase the page number to retrieve the next set of 100 filings. Example: 3 retrieves the third page. Default: 1",
        },
        "top_n_results": {
            "type": "integer",
            "description": "The top N results to return after the query. Useful if you are not sure the result you are loooking for is ranked first after your query.",
        },
    }
    required_arguments: list[str] = [
        "query",
        "start_date",
        "end_date",
        "top_n_results",
    ]

    def __init__(
        self,
        sec_api_key: str | None = None,
    ):
        super().__init__()
        if sec_api_key is None:
            sec_api_key = os.getenv("SEC_EDGAR_API_KEY")
        if not sec_api_key:
            raise ValueError("SEC_EDGAR_API_KEY is not set")
        self.sec_api_key: str = sec_api_key

        self.sec_api_url: str = "https://api.sec-api.io/full-text-search"

    @retry_on_429
    async def _execute_search(
        self,
        query: str,
        start_date: str,
        end_date: str,
        top_n_results: int,
        page: int | None = None,
        form_types: list[str] | str | None = None,
        ciks: list[str] | str | None = None,
    ) -> list[str]:
        """
        Search the EDGAR Database through the SEC API asynchronously.

        Args:
            query (str): The keyword or phrase to search
            form_types (list[str]): List of form types to search
            ciks (list[str]): List of CIKs to filter by
            start_date (str): Start date for the search range in yyyy-mm-dd format
            end_date (str): End date for the search range in yyyy-mm-dd format
            page (int): Pagination for results
            top_n_results (int): The top N results to return

        Returns:
            list[str]: A list of filing results
        """

        # Parse form_types if it's a string representation of a JSON array
        if (
            isinstance(form_types, str)
            and form_types.startswith("[")
            and form_types.endswith("]")
        ):
            try:
                form_types = json.loads(form_types.replace("'", '"'))
            except json.JSONDecodeError:
                # Fallback to simple parsing if JSON parsing fails
                form_types = [
                    item.strip(" \"'") for item in form_types[1:-1].split(",")
                ]

        # Parse ciks if it's a string representation of a JSON array
        if isinstance(ciks, str) and ciks.startswith("[") and ciks.endswith("]"):
            try:
                ciks = json.loads(ciks.replace("'", '"'))
            except json.JSONDecodeError:
                # Fallback to simple parsing if JSON parsing fails
                ciks = [item.strip(" \"'") for item in ciks[1:-1].split(",")]  #

        if end_date > MAX_END_DATE:
            end_date = MAX_END_DATE

        payload: dict[str, str | int | list[str]] = {
            "query": query,
            "startDate": start_date,
            "endDate": end_date,  # This will always be at most "2025-04-07"
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
            async with session.post(
                self.sec_api_url, json=payload, headers=headers
            ) as response:
                response.raise_for_status()  # This will raise ClientResponseError
                result = await response.json()

        return result.get("filings", [])[: int(top_n_results)]

    async def call_tool(
        self, arguments: dict[str, Any], data_storage: dict[str, Any], llm: LLM
    ) -> dict[str, Any]:
        try:
            return await self._execute_search(**arguments)
        except Exception as e:
            error_msh = f"SEC API error: {e}"
            if VERBOSE:
                error_msh += f"\nTraceback: {traceback.format_exc()}"
            tool_logger.error(error_msh)
            raise e


class ParseHtmlPage(Tool):
    name: str = "parse_html_page"
    description: str = """
        Parse an HTML page. This tool is used to parse the HTML content of a page and saves the content outside of the conversation to avoid context window issues.
        You should provide both the URL of the page to parse, as well as the key you want to use to save the result in the agent's data structure.
        The data structure is a dictionary.
    """.strip()

    input_arguments: dict[str, Any] = {
        "url": {"type": "string", "description": "The URL of the HTML page to parse"},
        "key": {
            "type": "string",
            "description": "The key to use when saving the result in the conversation's data structure (dict).",
        },
    }
    required_arguments: list[str] = ["url", "key"]

    def __init__(self):
        super().__init__()

    @retry_on_429
    async def _parse_html_page(self, url: str) -> str:
        """
        Helper method to parse an HTML page and extract its text content.

        Args:
            url (str): The URL of the HTML page to parse

        Returns:
            str: The parsed text content
        """
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
                else:
                    is_verbose = os.environ.get("EDGAR_AGENT_VERBOSE", "0") == "1"
                    if is_verbose:
                        raise Exception(
                            str(e) + "\nTraceback: " + traceback.format_exc()
                        )
                    else:
                        raise Exception(str(e))

        soup = BeautifulSoup(html_content, "html.parser")

        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            _ = script_or_style.extract()

        # Get text
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)

        return text

    async def _save_tool_output(
        self, output: list[str], key: str, data_storage: dict[str, Any]
    ) -> str:
        """
        Save the parsed HTML text to the data_storage dictionary.

        Args:
            output (list[str]): The parsed text output from call_tool
            data_storage (dict): The dictionary to save the results to
        """
        if not output:
            raise ValueError("HTML output was empty")

        tool_result = ""
        if key in data_storage:
            tool_result = "WARNING: The key already exists in the data storage. The new result overwrites the old one.\n"
        tool_result += (
            f"SUCCESS: The result has been saved to the data storage under the key: {key}."
            + "\n"
        )

        data_storage[key] = output

        keys_list = "\n".join(data_storage.keys())
        tool_result += (
            f"""
        The data_storage currently contains the following keys:
        {keys_list}
        """.strip()
            + "\n"
        )

        return tool_result

    @override
    async def call_tool(
        self, arguments: dict[str, Any], data_storage: dict[str, Any], llm: LLM
    ) -> dict[str, Any]:
        """
        Parse an HTML page and return its text content.

        Args:
            arguments (dict): Dictionary containing 'url' and 'key'

        Returns:
            list[str]: A list containing the parsed text
        """
        url = arguments.get("url", "")
        if not url:
            raise ValueError("URL is required")
        key = arguments.get("key", "")
        if not key:
            raise ValueError("Key is required")

        text_output = await self._parse_html_page(url)
        tool_result = await self._save_tool_output(text_output, key, data_storage)

        return tool_result  # pyright: ignore


class RetrieveInformation(Tool):
    name: str = "retrieve_information"
    description: str = """
    Retrieve information from the conversation's data structure (dict) and allow character range extraction.
    
    IMPORTANT: Your prompt MUST include at least one key from the data storage using the exact format: {{key_name}}
    
    For example, if you want to analyze data stored under the key "financial_report", your prompt should look like:
    "Analyze the following financial report and extract the revenue figures: {{financial_report}}"
    
    The {{key_name}} will be replaced with the actual content stored under that key before being sent to the LLM.
    If you don't use this exact format with double braces, the tool will fail to retrieve the information.
    
    You can optionally specify character ranges for each document key to extract only portions of documents. That can be useful to avoid token limit errors or improve efficiency by selecting only part of the document.
    For example, if "financial_report" contains "Annual Report 2023" and you specify a range [1, 5] for that key,
    only "nnual" will be inserted into the prompt.
    
    The output is the result from the LLM that receives the prompt with the inserted data.
    """.strip()
    input_arguments: dict[str, Any] = {
        "prompt": {
            "type": "string",
            "description": "The prompt that will be passed to the LLM. You MUST include at least one data storage key in the format {{key_name}} - for example: 'Summarize this 10-K filing: {{company_10k}}'. The content stored under each key will replace the {{key_name}} placeholder.",
        },
        "input_character_ranges": {
            "type": "object",
            "description": "A dictionary mapping document keys to their character ranges. Each range should be an array where the first element is the start index and the second element is the end index. Can be used to only read portions of documents. By default, the full document is used. To use the full document, set the range to an empty list [].",
            "additionalProperties": {
                "type": "array",
                "items": {
                    "type": "integer",
                },
            },
        },
    }
    required_arguments: list[str] = ["prompt"]

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
        )

    async def call_tool(
        self, arguments: dict[str, Any], data_storage: dict[str, Any], llm: LLM
    ) -> dict[str, Any]:
        prompt: str = arguments["prompt"]
        input_character_ranges = arguments.get("input_character_ranges", {})
        if input_character_ranges is None:
            input_character_ranges = {}

        # Verify that the prompt contains at least one placeholder in the correct format
        if not re.search(r"{{[^{}]+}}", prompt):
            raise ValueError(
                "ERROR: Your prompt must include at least one key from data storage in the format {{key_name}}. Please try again with the correct format."
            )

        # Find all keys in the prompt
        keys = re.findall(r"{{([^{}]+)}}", prompt)
        formatted_data = {}

        # Apply character range to each document before substitution
        for key in keys:
            if key not in data_storage:
                raise KeyError(
                    f"ERROR: The key '{key}' was not found in the data storage. Available keys are: {', '.join(data_storage.keys())}"
                )

            # Extract the specified character range from the document if provided
            doc_content = data_storage[key]

            if key in input_character_ranges:
                char_range = input_character_ranges[key]
                if len(char_range) == 0:
                    formatted_data[key] = doc_content
                elif len(char_range) != 2:
                    raise ValueError(
                        f"ERROR: The character range for key '{key}' must be an list with two elements or an empty list. Please try again with the correct format."
                    )
                else:
                    start_idx = int(char_range[0])
                    end_idx = int(char_range[1])
                    formatted_data[key] = doc_content[start_idx:end_idx]
            else:
                # Use the full document if no range is specified
                formatted_data[key] = doc_content

        # Convert {{key}} format to Python string formatting
        formatted_prompt = re.sub(r"{{([^{}]+)}}", r"{\1}", prompt)

        try:
            prompt = formatted_prompt.format(**formatted_data)
        except KeyError as e:
            raise KeyError(
                f"ERROR: The key {str(e)} was not found in the data storage. Available keys are: {', '.join(data_storage.keys())}"
            )

        response = await llm.query(prompt)

        return {
            "retrieval": response.output_text_str,
            "usage": response.metadata,
        }
