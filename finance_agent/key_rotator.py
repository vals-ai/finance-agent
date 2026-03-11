import itertools
import os


class KeyRotator:
    """Round-robin API key rotator for distributing requests across multiple keys."""

    def __init__(self, keys: list[str]) -> None:
        if not keys:
            raise ValueError("At least one key must be provided")
        self._keys = keys
        self._cycle = itertools.cycle(keys)

    @classmethod
    def from_env(cls, env_var: str) -> "KeyRotator":
        raw = os.getenv(env_var)
        if not raw:
            raise ValueError(f"{env_var} is not set")
        keys = [k.strip() for k in raw.split(";") if k.strip()]
        if not keys:
            raise ValueError(f"{env_var} is empty after parsing")
        return cls(keys)

    def next_key(self) -> str:
        return next(self._cycle)

    @property
    def key_count(self) -> int:
        return len(self._keys)


_rotators: dict[str, KeyRotator] = {}


def get_rotator(env_var: str) -> KeyRotator:
    """Returns a shared KeyRotator per env var name, creating on first access."""
    if env_var not in _rotators:
        _rotators[env_var] = KeyRotator.from_env(env_var)
    return _rotators[env_var]
