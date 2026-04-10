import logging
from collections.abc import Callable
from typing import Any

import aiohttp
import backoff

logger = logging.getLogger(__name__)


def _get_http_status(exception: Exception) -> int | None:
    if isinstance(exception, aiohttp.ClientResponseError):
        return exception.status

    return None


def _get_request_url(exception: Exception) -> str | None:
    if isinstance(exception, aiohttp.ClientResponseError):
        return str(exception.request_info.real_url)

    return None


def retry_http_errors(
    *status_codes: int,
    url_patterns: dict[int, list[str]] | None = None,
) -> Callable:
    """Retry on specific HTTP status codes with exponential backoff.

    Args:
        status_codes: Status codes to always retry on.
        url_patterns: Optional mapping of status code -> list of domain substrings.
            When provided, that status code is only retried if the request URL
            contains one of the specified substrings.
    """

    def should_retry(exception: Exception) -> bool:
        status = _get_http_status(exception)
        if status is None:
            return False

        if status in status_codes:
            logger.error(f"{status} error: {exception}")
            return True

        if url_patterns and status in url_patterns:
            url = _get_request_url(exception)
            if url and any(pattern in url for pattern in url_patterns[status]):
                logger.error(f"{status} error: {exception}")
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
