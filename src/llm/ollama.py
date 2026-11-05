"""Talk to a local Ollama server through its /api/chat endpoint.

Settings come from the environment (or the project's .env file):
OLLAMA_HOST (default http://127.0.0.1:11434) and OLLAMA_MODEL (default llama3.2:3b).
"""

import json
import os
from collections.abc import Iterator
from pathlib import Path

import httpx
from dotenv import load_dotenv

from .base import LLMError, LLMUnavailableError, ModelNotFoundError

DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "llama3.2:3b"

# Ollama's default context window is small and silently cuts off the start of
# a long prompt, which would drop sources without any warning. Set it explicitly.
NUM_CTX = 4096
TEMPERATURE = 0.1  # low, so answers stick to the sources

# Generating on a CPU is slow, so wait a long time for the reply.
TIMEOUT = httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=10.0)

_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class OllamaLLM:
    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        num_ctx: int = NUM_CTX,
        temperature: float = TEMPERATURE,
        client: httpx.Client | None = None,
    ):
        load_dotenv(_ENV_FILE)  # variables already set in the environment win
        self.host = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST).rstrip("/")
        self.model = model or os.environ.get("OLLAMA_MODEL") or DEFAULT_MODEL
        self.num_ctx = num_ctx
        self.temperature = temperature
        # A client can be passed in for testing (httpx.MockTransport).
        self._client = client or httpx.Client(timeout=TIMEOUT)

    # ------------------------------------------------------------ requests

    def _payload(self, system: str, user: str, stream: bool) -> dict:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": stream,
            "options": {"num_ctx": self.num_ctx, "temperature": self.temperature},
        }

    def _unreachable(self) -> LLMUnavailableError:
        return LLMUnavailableError(
            f"Could not reach Ollama at {self.host}. "
            "Start Ollama (open the Ollama app, or run: ollama serve) and try again."
        )

    def _timed_out(self) -> LLMError:
        return LLMError(
            f"Ollama at {self.host} took too long to answer. "
            "The model may still be loading; try again in a moment."
        )

    def _check_status(self, response: httpx.Response) -> None:
        """Turn an HTTP error from Ollama into a friendly exception."""
        if response.status_code < 400:
            return
        message = ""
        try:
            message = str(response.json().get("error", ""))
        except (ValueError, AttributeError):
            pass
        if response.status_code == 404 and "not found" in message.lower():
            raise ModelNotFoundError(
                f"Model {self.model!r} is not installed in Ollama. Run: ollama pull {self.model}"
            )
        raise LLMError(f"Ollama returned an error ({response.status_code}): {message or 'no details'}")

    # ----------------------------------------------------------- the LLM API

    def generate(self, system: str, user: str) -> str:
        try:
            response = self._client.post(
                f"{self.host}/api/chat", json=self._payload(system, user, stream=False)
            )
        except httpx.ConnectError as error:
            raise self._unreachable() from error
        except httpx.TimeoutException as error:
            raise self._timed_out() from error
        self._check_status(response)
        try:
            return response.json()["message"]["content"]
        except (ValueError, KeyError, TypeError) as error:
            raise LLMError("Ollama sent a reply in an unexpected format.") from error

    def stream(self, system: str, user: str) -> Iterator[str]:
        """Yield the reply piece by piece (Ollama sends one JSON object per line)."""
        try:
            with self._client.stream(
                "POST", f"{self.host}/api/chat", json=self._payload(system, user, stream=True)
            ) as response:
                if response.status_code >= 400:
                    response.read()  # the error body must be read before it can be parsed
                    self._check_status(response)
                for line in response.iter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except ValueError as error:
                        raise LLMError("Ollama sent a reply in an unexpected format.") from error
                    if "error" in data:
                        raise LLMError(f"Ollama reported an error: {data['error']}")
                    piece = (data.get("message") or {}).get("content", "")
                    if piece:
                        yield piece
                    if data.get("done"):
                        return
        except httpx.ConnectError as error:
            raise self._unreachable() from error
        except httpx.TimeoutException as error:
            raise self._timed_out() from error
