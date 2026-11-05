from collections.abc import Iterator
from typing import Protocol


class LLMError(Exception):
    """A problem talking to the language model, with a message fit to show a user."""


class LLMUnavailableError(LLMError):
    """The model server cannot be reached."""


class ModelNotFoundError(LLMError):
    """The server is running but does not have the requested model."""


class LLM(Protocol):
    """Anything that can turn a system prompt and a user message into text."""

    def generate(self, system: str, user: str) -> str:
        """The complete reply."""
        ...

    def stream(self, system: str, user: str) -> Iterator[str]:
        """The reply as it is produced, a few characters at a time."""
        ...
