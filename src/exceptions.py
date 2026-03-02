from collections.abc import Callable
from typing import Any

import aiohttp
import backoff
from logger import get_logger

exceptions_logger = get_logger(__name__)


def retry_http_errors(*status_codes: int) -> Callable:
    def should_retry(exception: Exception) -> bool:
        for code in status_codes:
            if isinstance(exception, aiohttp.ClientResponseError) and exception.status == code or str(code) in str(exception):
                exceptions_logger.error(f"{code} error: {exception}")
                return True
        return False

    def decorator(func: Callable) -> Callable:
        @backoff.on_exception(
            backoff.expo,
            aiohttp.ClientResponseError,
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
