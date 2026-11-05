import pytest

from src.llm.base import LLMUnavailableError, ModelNotFoundError
from src.llm.ollama import OllamaLLM


@pytest.mark.integration
def test_real_ollama_answers_a_trivial_question():
    llm = OllamaLLM()
    try:
        reply = llm.generate("Answer with a single word.", "What colour is the sky on a clear day?")
    except LLMUnavailableError:
        pytest.skip(f"Ollama is not running at {llm.host}; start it to run this test.")
    except ModelNotFoundError:
        pytest.skip(f"Model {llm.model} is not installed; run: ollama pull {llm.model}")
    assert isinstance(reply, str) and reply.strip()
