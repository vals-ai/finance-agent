import json
import backoff
import aiohttp
import re
import os

from abc import ABC, abstractmethod
from bs4 import BeautifulSoup
from openai.types.chat import ChatCompletionMessageToolCall
from anthropic.types.tool_use_block import ToolUseBlock

from .llm import GeneralLLM
from .logger import get_logger

tool_logger = get_logger(__name__)


def is_429(exception):
    is429 = (
        isinstance(exception, aiohttp.ClientResponseError)
        and exception.status == 429
        or "429" in str(exception)
    )
    if is429:
        print(f"429 error: {exception}")
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
    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

    return wrapper


class Tool(ABC):
    """
    Abstract base class for tools.
    """

    name: str
    description: str
    input_arguments: dict
    required_arguments: list[str]

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__()

    def parse_tool_message(
        self,
        provider="openai",
        message: str | ChatCompletionMessageToolCall | ToolUseBlock = None,
    ):
        """
        Get the tool format for different providers.

        Args:
            provider (str): The provider to format the tool for ('openai' or 'anthropic')

        Returns:
            dict: Formatted tool definition
        """
        if provider.lower() != "anthropic":
            arguments = message.function.arguments
        elif provider.lower() == "anthropic":
            arguments = message.input
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        return arguments

    def get_tool_json(self, provider: str = "openai", strict: bool = True) -> dict:
        if provider.lower() == "anthropic":
            return {
                "name": self.name,
                "description": self.description,
                "input_schema": {
                    "type": "object",
                    "properties": self.input_arguments,
                    "required": self.required_arguments,
                },
            }
        elif provider.lower() == "mistralai":
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": {
                        "type": "object",
                        "properties": self.input_arguments,
                        "required": self.required_arguments,
                    },
                },
            }
        else:
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": {
                        "type": "object",
                        "properties": self.input_arguments,
                        "required": self.required_arguments,
                        "additionalProperties": False,
                    },
                    "strict": strict,
                },
            }

    @abstractmethod
    def call_tool(self, arguments: dict, *args, **kwargs) -> list[str]:
        pass

    async def __call__(self, arguments: dict = None, *args, **kwargs) -> list[str]:
        tool_logger.info(
            f"\033[1;34m[TOOL: {self.name.upper()}]\033[0m Calling with arguments: {arguments}"
        )

        try:
            tool_result = await self.call_tool(arguments, *args, **kwargs)
            tool_logger.info(
                f"\033[1;32m[TOOL: {self.name.upper()}]\033[0m Returned: {tool_result}"
            )
            if self.name == "retrieve_information":
                return {
                    "success": True,
                    "result": tool_result["retrieval"],
                    "usage": tool_result["usage"],
                }
            else:
                return {"success": True, "result": json.dumps(tool_result)}
        except Exception as e:
            tool_logger.error(
                f"\033[1;31m[TOOL: {self.name.upper()}]\033[0m Error: {e}"
            )
            return {"success": False, "result": str(e)}


class GoogleWebSearch(Tool):
    name: str = "google_web_search"
    description: str = "Search the web for information"
    input_arguments: dict = {
        "search_query": {
            "type": "string",
            "description": "The query to search for",
        }
    }
    required_arguments: list[str] = ["search_query"]

    def __init__(
        self,
        top_n_results: int = 10,
        serpapi_api_key: str = os.getenv("SERPAPI_API_KEY"),
        *args,
        **kwargs,
    ):
        super().__init__(
            self.name,
            self.description,
            self.input_arguments,
            self.required_arguments,
            *args,
            **kwargs,
        )
        self.top_n_results = top_n_results
        self.serpapi_api_key = serpapi_api_key

    @retry_on_429
    async def _execute_search(self, search_query: str) -> list[str]:
        """
        Search the web for information using Google Search.

        Args:
            search_query (str): The query to search for

        Returns:
            list[str]: A list of results from Google Search
        """
        params = {
            "api_key": self.serpapi_api_key,
            "engine": "google",
            "q": search_query,
            "num": self.top_n_results,
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://serpapi.com/search.json", params=params
            ) as response:
                response.raise_for_status()  # This will raise ClientResponseError
                results = await response.json()

        return results.get("organic_results", [])

    async def call_tool(self, arguments: dict) -> list[str]:
        results = await self._execute_search(**arguments)
        return results


class EDGARSearch(Tool):
    name: str = "edgar_search"
    description: str = (
        """
    Search the EDGAR Database through the SEC API.
    You should provide a query, a list of form types, a list of CIKs, a start date, an end date, a page number, and a top N results.
    The results are returned as a list of dictionaries, each containing the metadata for a filing. It does not contain the full text of the filing.
    """.strip()
    )
    input_arguments: dict = {
        "query": {
            "type": "string",
            "description": "The keyword or phrase to search, such as 'substantial doubt' OR 'material weakness'",
        },
        "form_types": {
            "type": "array",
            "description": "Limits search to specific SEC form types (e.g., ['8-K', '10-Q']) list of strings. Default is None (all form types)",
            "items": {"type": "string"},
        },
        "ciks": {
            "type": "array",
            "description": "Filters results to filings by specified CIKs, type list of strings. Default is None (all filers).",
            "items": {"type": "string"},
        },
        "start_date": {
            "type": "string",
            "description": "Start date for the search range in yyyy-mm-dd format. Used with endDate to define the date range. Example: '2024-01-01'. Default is 30 days ago",
        },
        "end_date": {
            "type": "string",
            "description": "End date for the search range, in the same format as startDate. Default is today",
        },
        "page": {
            "type": "string",
            "description": "Pagination for results. Default is '1'",
        },
        "top_n_results": {
            "type": "integer",
            "description": "The top N results to return after the query. Useful if you are not sure the result you are loooking for is ranked first after your query.",
        },
    }
    required_arguments: list[str] = [
        "query",
        "form_types",
        "ciks",
        "start_date",
        "end_date",
        "page",
        "top_n_results",
    ]

    def __init__(
        self,
        sec_api_key: str = os.getenv("SEC_EDGAR_API_KEY"),
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )
        self.sec_api_key = sec_api_key
        print
        self.sec_api_url = "https://api.sec-api.io/full-text-search"

    @retry_on_429
    async def _execute_search(
        self,
        query: str,
        form_types: list[str],
        ciks: list[str],
        start_date: str,
        end_date: str,
        page: int,
        top_n_results: int,
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
                ciks = [item.strip(" \"'") for item in ciks[1:-1].split(",")]

        payload = {
            "query": query,
            "formTypes": form_types,
            "ciks": ciks,
            "startDate": start_date,
            "endDate": end_date,
            "page": page,
        }

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

    async def call_tool(self, arguments: dict) -> list[str]:
        try:
            return await self._execute_search(**arguments)
        except Exception as e:
            tool_logger.error(f"SEC API error: {e}")
            raise


class ParseHtmlPage(Tool):
    name: str = "parse_html_page"
    description: str = (
        """
        Parse an HTML page. This tool is used to parse the HTML content of a page and saves the content outside of the conversation to avoid context window issues.
        You should provide both the URL of the page to parse, as well as the key you want to use to save the result in the agent's data structure.
        The data structure is a dictionary.
    """.strip()
    )

    input_arguments: dict = {
        "url": {"type": "string", "description": "The URL of the HTML page to parse"},
        "key": {
            "type": "string",
            "description": "The key to use when saving the result in the conversation's data structure (dict).",
        },
    }
    required_arguments: list[str] = ["url", "key"]

    def __init__(
        self, headers: dict = {"User-Agent": "ValsAI/antoine@vals.ai"}, *args, **kwargs
    ):
        super().__init__(
            *args,
            **kwargs,
        )
        self.headers = headers

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
                    url, headers=self.headers, timeout=60
                ) as response:
                    response.raise_for_status()
                    html_content = await response.text()
            except Exception as e:
                if len(str(e)) == 0:
                    raise TimeoutError(
                        "Timeout error when parsing HTML page after 60 seconds. The URL might be blocked or the server is taking too long to respond."
                    )
                else:
                    raise e

        soup = BeautifulSoup(html_content, "html.parser")

        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.extract()

        # Get text
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)

        return text

    async def _save_tool_output(
        self, output: list[str], key: str, data_storage: dict
    ) -> None:
        """
        Save the parsed HTML text to the data_storage dictionary.

        Args:
            output (list[str]): The parsed text output from call_tool
            data_storage (dict): The dictionary to save the results to
        """
        if not output:
            return

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

    async def call_tool(self, arguments: dict, data_storage: dict) -> list[str]:
        """
        Parse an HTML page and return its text content.

        Args:
            arguments (dict): Dictionary containing 'url' and 'key'

        Returns:
            list[str]: A list containing the parsed text
        """
        url = arguments.get("url")
        key = arguments.get("key")
        text_output = await self._parse_html_page(url)
        tool_result = await self._save_tool_output(text_output, key, data_storage)

        return tool_result


class RetrieveInformation(Tool):
    name: str = "retrieve_information"
    description: str = (
        """
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
    )
    input_arguments: dict = {
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
        self, arguments: dict, data_storage: dict, model: GeneralLLM, *args, **kwargs
    ) -> list[str]:
        prompt: str = arguments.get("prompt")
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

        model_response = await model.safe_chat(
            messages=[
                {"role": "user", "content": prompt},
            ],
            ignore_token_error=True,
        )

        return {
            "retrieval": model.parse_response(model_response),
            "usage": model.convert_usage(model_response.usage),
        }
