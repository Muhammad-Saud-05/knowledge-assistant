import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request

from .config import PROJECT_ROOT, Settings
from .deps import get_pool, get_settings
from .observability import request_id_of
from .schemas import DeleteResult, DocumentOut
from .uploads import uploaded_file_for

router = APIRouter()
logger = logging.getLogger("kb.api")

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


@router.delete("/documents/{document_id}", response_model=DeleteResult)
def delete_document(
    document_id: int,
    request: Request,
    pool=Depends(get_pool),
    settings: Settings = Depends(get_settings),
) -> DeleteResult:
    """Delete a document; its chunks and vectors go with it (ON DELETE CASCADE).

    If the document came from an upload, that uploaded copy is deleted from disk too.
    Files anywhere else (such as sample_corpus/) are never touched.
    """
    with pool.connection() as conn:
        row = conn.execute("SELECT path FROM documents WHERE id = %s", (document_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="That document does not exist.")
        stored_path = row[0]
        conn.execute("DELETE FROM documents WHERE id = %s", (document_id,))

    file_removed = False
    uploaded = uploaded_file_for(stored_path, settings.upload_dir, Path.cwd(), PROJECT_ROOT)
    if uploaded is not None:
        try:
            uploaded.unlink()
            file_removed = True
        except OSError:
            logger.warning("delete id=%s: could not remove the uploaded file", request_id_of(request))

    logger.info("delete id=%s document=%d file_removed=%s", request_id_of(request), document_id, file_removed)
    return DeleteResult(deleted=True, file_removed=file_removed)
