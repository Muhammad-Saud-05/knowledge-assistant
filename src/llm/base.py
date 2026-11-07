from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol


class LLMError(Exception):
    """A problem talking to the language model, with a message fit to show a user."""


class LLMUnavailableError(LLMError):
    """The model server cannot be reached."""


class ModelNotFoundError(LLMError):
    """The server is running but does not have the requested model."""


@dataclass
class ModelStatus:
    """Whether the model server is up and has the configured model."""

    reachable: bool
    model: str
    model_installed: bool | None  # None when the server could not be reached
    detail: str | None = None  # a short explanation when something is wrong


class LLM(Protocol):
    """Anything that can turn a system prompt and a user message into text."""

    def generate(self, system: str, user: str) -> str:
        """The complete reply."""
        ...

    def stream(self, system: str, user: str) -> Iterator[str]:
        """The reply as it is produced, a few characters at a time."""
        ...

    def status(self) -> ModelStatus:
        """A quick check that the server is up and the model is installed (never raises)."""
        ...

    def warm_up(self) -> None:
        """Load the model into memory ahead of the first question."""
        ...
