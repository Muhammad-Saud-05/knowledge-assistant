"""Talk to a local Ollama server through its /api/chat endpoint.

Settings come from the environment (or the project's .env file):
OLLAMA_HOST (default http://127.0.0.1:11434), OLLAMA_MODEL (default llama3.2:3b) and
OLLAMA_KEEP_ALIVE (default 30m, how long Ollama keeps the model loaded after a request).
"""

import json
import os
from collections.abc import Iterator
from pathlib import Path

import httpx
from dotenv import load_dotenv

from .base import LLMError, LLMUnavailableError, ModelNotFoundError, ModelStatus

DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "llama3.2:3b"
# Ollama unloads a model after 5 idle minutes by default, and loading it again takes
# many seconds on a CPU. Ask it to keep the model in memory for longer.
DEFAULT_KEEP_ALIVE = "30m"

# Ollama's default context window is small and silently cuts off the start of
# a long prompt, which would drop sources without any warning. Set it explicitly.
NUM_CTX = 4096
TEMPERATURE = 0.1  # low, so answers stick to the sources

# A status check must answer quickly, whatever the main timeout is.
STATUS_TIMEOUT = 3.0

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
        keep_alive: str | None = None,
    ):
        load_dotenv(_ENV_FILE)  # variables already set in the environment win
        self.host = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST).rstrip("/")
        self.model = model or os.environ.get("OLLAMA_MODEL") or DEFAULT_MODEL
        self.keep_alive = keep_alive or os.environ.get("OLLAMA_KEEP_ALIVE") or DEFAULT_KEEP_ALIVE
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
            "keep_alive": self.keep_alive,
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

    def status(self) -> ModelStatus:
        """Is Ollama running, and is the configured model installed? Never raises."""
        try:
            response = self._client.get(f"{self.host}/api/tags", timeout=STATUS_TIMEOUT)
            response.raise_for_status()
            names = [m.get("name", "") for m in response.json().get("models", [])]
        except httpx.HTTPStatusError:
            return ModelStatus(False, self.model, None, "Ollama returned an error.")
        except (httpx.HTTPError, ValueError, AttributeError):
            return ModelStatus(False, self.model, None, "Ollama isn't running.")
        # "llama3.2" means "llama3.2:latest" to Ollama.
        wanted = {self.model, f"{self.model}:latest"}
        installed = any(name in wanted for name in names)
        detail = None if installed else f"The model {self.model} isn't installed (run: ollama pull {self.model})."
        return ModelStatus(True, self.model, installed, detail)

    def warm_up(self) -> None:
        """Load the model into memory without generating anything.

        Ollama loads a model when it gets a chat request with no messages. Raises
        the same friendly errors as generate().
        """
        payload = {"model": self.model, "messages": [], "stream": False, "keep_alive": self.keep_alive}
        try:
            response = self._client.post(f"{self.host}/api/chat", json=payload)
        except httpx.ConnectError as error:
            raise self._unreachable() from error
        except httpx.TimeoutException as error:
            raise self._timed_out() from error
        self._check_status(response)

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
