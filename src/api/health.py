from fastapi import APIRouter, Depends

from .deps import get_llm, get_pool
from .schemas import DatabaseHealth, HealthResponse, OllamaHealth

router = APIRouter()


def check_database(pool) -> DatabaseHealth:
    try:
        with pool.connection() as conn:
            documents, chunks = conn.execute(
                "SELECT (SELECT count(*) FROM documents), (SELECT count(*) FROM chunks)"
            ).fetchone()
        return DatabaseHealth(ok=True, documents=documents, chunks=chunks)
    except Exception:  # noqa: BLE001 - any failure just means "down"; details could include host names
        return DatabaseHealth(ok=False, detail="The database isn't reachable.")


def check_ollama(llm) -> OllamaHealth:
    try:
        status = llm.status()
        return OllamaHealth(
            ok=status.reachable,
            model=status.model,
            model_installed=status.model_installed,
            detail=status.detail,
        )
    except Exception:  # noqa: BLE001
        return OllamaHealth(ok=False, model=getattr(llm, "model", "unknown"), detail="Ollama isn't running.")


@router.get("/health", response_model=HealthResponse)
def health(pool=Depends(get_pool), llm=Depends(get_llm)) -> HealthResponse:
    """Always answers 200; the parts that are down are marked instead of crashing the request."""
    database = check_database(pool)
    ollama = check_ollama(llm)
    healthy = database.ok and ollama.ok and ollama.model_installed is not False
    return HealthResponse(status="ok" if healthy else "degraded", database=database, ollama=ollama)
