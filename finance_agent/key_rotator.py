import asyncio
import itertools
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


DEFAULT_MAX_CONCURRENT = int(os.getenv("KEY_ROTATOR_MAX_CONCURRENT", "50"))


class KeyRotator:
    """Round-robin API key rotator with concurrency control."""

    def __init__(self, keys: list[str], max_concurrent: int = DEFAULT_MAX_CONCURRENT) -> None:
        if not keys:
            raise ValueError("At least one key must be provided")
        self._keys = keys
        self._cycle = itertools.cycle(keys)
        self._semaphore = asyncio.Semaphore(max_concurrent)

    @classmethod
    def from_env(cls, env_var: str, max_concurrent: int = DEFAULT_MAX_CONCURRENT) -> "KeyRotator":
        raw = os.getenv(env_var)
        if not raw:
            raise ValueError(f"{env_var} is not set")
        keys = [k.strip() for k in raw.split(";") if k.strip()]
        if not keys:
            raise ValueError(f"{env_var} is empty after parsing")
        return cls(keys, max_concurrent)

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[str]:
        """Acquire a semaphore slot and yield the next key. Holds the slot until the caller's async block exits."""
        async with self._semaphore:
            yield next(self._cycle)

    @property
    def key_count(self) -> int:
        return len(self._keys)


_rotators: dict[str, KeyRotator] = {}


def get_rotator(env_var: str, max_concurrent: int = DEFAULT_MAX_CONCURRENT) -> KeyRotator:
    """Returns a shared KeyRotator per env var name, creating on first access."""
    if env_var not in _rotators:
        _rotators[env_var] = KeyRotator.from_env(env_var, max_concurrent)
    return _rotators[env_var]
