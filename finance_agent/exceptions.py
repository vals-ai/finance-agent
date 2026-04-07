import logging
from collections.abc import Callable
from typing import Any

import aiohttp
import backoff
import httpx

logger = logging.getLogger(__name__)


def retry_http_errors(*status_codes: int | tuple[int, str]) -> Callable:
    """Retry on specific HTTP status codes. Each entry can be:
    - an int: retry unconditionally on that status code
    - a tuple (int, str): retry only if the URL contains the given string
    """

    parsed = []
    for entry in status_codes:
        if isinstance(entry, int):
            parsed.append((entry, None))
        else:
            parsed.append((entry[0], entry[1]))

    def _url_from_exception(exception: Exception) -> str:
        if isinstance(exception, aiohttp.ClientResponseError) and exception.request_info:
            return str(exception.request_info.url)
        if isinstance(exception, httpx.HTTPStatusError):
            return str(exception.request.url)
        return ""

    def should_retry(exception: Exception) -> bool:
        for code, url_pattern in parsed:
            if (
                isinstance(exception, aiohttp.ClientResponseError)
                and exception.status == code
                or isinstance(exception, httpx.HTTPStatusError)
                and exception.response.status_code == code
                or str(code) in str(exception)
            ):
                if url_pattern and url_pattern not in _url_from_exception(exception):
                    continue
                logger.error(f"{code} error: {exception}")
                return True
        return False

    def decorator(func: Callable) -> Callable:
        @backoff.on_exception(
            backoff.expo,
            Exception,
            max_tries=100,
            max_value=120,
            base=2,
            factor=3,
            jitter=backoff.full_jitter,
            giveup=lambda e: not should_retry(e),
        )
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await func(*args, **kwargs)

        return wrapper

    return decorator
