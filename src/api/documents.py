from fastapi import APIRouter, Depends

from .deps import get_pool
from .schemas import DocumentOut

router = APIRouter()

_LIST_SQL = """
    SELECT d.id, d.title, d.path, d.source_type, d.doc_date, d.ingested_at, count(c.id) AS chunks
    FROM documents d
    LEFT JOIN chunks c ON c.document_id = d.id
    GROUP BY d.id
    ORDER BY d.ingested_at DESC, d.id DESC
"""


@router.get("/documents", response_model=list[DocumentOut])
def list_documents(pool=Depends(get_pool)) -> list[DocumentOut]:
    """Every stored document with its number of chunks, newest first."""
    with pool.connection() as conn:
        rows = conn.execute(_LIST_SQL).fetchall()
    return [
        DocumentOut(
            id=id_, title=title, path=path, source_type=source_type,
            doc_date=doc_date, ingested_at=ingested_at, chunks=chunks,
        )
        for id_, title, path, source_type, doc_date, ingested_at, chunks in rows
    ]
