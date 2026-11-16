import logging
import re
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi import Path as PathParam

from .config import PROJECT_ROOT, Settings
from .deps import get_pool, get_settings
from .observability import request_id_of
from .schemas import DocumentOut
from .uploads import uploaded_file_for

router = APIRouter()
logger = logging.getLogger("kb.api")

MAX_DOCUMENT_ID = 2**31 - 1  # the database's integer limit

_LIST_SQL = """
    SELECT d.id, d.title, d.path, d.source_type, d.doc_date, d.ingested_at, count(c.id) AS chunk_count
    FROM documents d
    LEFT JOIN chunks c ON c.document_id = d.id
    GROUP BY d.id
    ORDER BY d.ingested_at DESC, d.id DESC
"""


def display_name_of(stored_path: str) -> str:
    """The file name part of a stored path, whichever kind of slash it uses. Never a folder."""
    name = re.split(r"[\\/]", stored_path.rstrip("/\\"))[-1]
    return name or "document"


@router.get("/documents", response_model=list[DocumentOut])
def list_documents(pool=Depends(get_pool)) -> list[DocumentOut]:
    """Every stored document with its number of chunks, newest first. Paths are never returned."""
    with pool.connection() as conn:
        rows = conn.execute(_LIST_SQL).fetchall()
    return [
        DocumentOut(
            id=id_, title=title, source_type=source_type, display_name=display_name_of(path),
            doc_date=doc_date, ingested_at=ingested_at, chunk_count=chunk_count,
        )
        for id_, title, path, source_type, doc_date, ingested_at, chunk_count in rows
    ]


@router.delete("/documents/{document_id}", status_code=204)
def delete_document(
    request: Request,
    document_id: int = PathParam(ge=1, le=MAX_DOCUMENT_ID),
    pool=Depends(get_pool),
    settings: Settings = Depends(get_settings),
) -> Response:
    """Delete a document; its chunks and vectors go with it (ON DELETE CASCADE). 204 on success, 404 if
    there is no such document.

    If the document came from an upload, that uploaded copy is deleted from disk too, but only after
    its resolved path is checked to lie inside the upload folder. Files anywhere else (such as the
    sample or corpus folders) are never touched.
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

    # Ids and outcomes only: no document text and no paths.
    logger.info("delete id=%s document=%d file_removed=%s", request_id_of(request), document_id, file_removed)
    return Response(status_code=204)
