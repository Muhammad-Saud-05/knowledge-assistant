import logging
import os
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from ..ingestion.pipeline import ingest_file, normalise_path
from .config import Settings
from .deps import get_embedder, get_pool, get_settings
from .observability import request_id_of
from .schemas import IngestResult
from .uploads import InvalidUpload, destination_for, sanitize_filename

router = APIRouter()
logger = logging.getLogger("kb.api")

READ_CHUNK_BYTES = 1024 * 1024
# Only this kind of failure message is passed on to the user. Anything else could mention
# server paths or file content, so it is replaced by a generic sentence.
_SAFE_FAILURE_MARKER = "no chunk has at least"
GENERIC_FAILURE = "The file could not be read or stored. Is it a valid, text-based PDF or Markdown file?"


async def read_limited(file: UploadFile, max_bytes: int) -> bytes:
    """Read the upload, stopping as soon as it goes over the size limit."""
    data = bytearray()
    while True:
        chunk = await file.read(READ_CHUNK_BYTES)
        if not chunk:
            break
        data.extend(chunk)
        if len(data) > max_bytes:
            raise InvalidUpload(f"The file is larger than the {max_bytes // (1024 * 1024)} MB limit.", 413)
    if not data:
        raise InvalidUpload("The file is empty.")
    return bytes(data)


def store_and_ingest(data: bytes, safe_name: str, settings: Settings, pool, embedder) -> IngestResult:
    """Save the upload, then ingest it. Runs in a worker thread.

    The new bytes are written to a temporary name and swapped in, with the old file kept
    as a backup until ingestion succeeds, so a failed update never loses a good file.
    """
    destination = destination_for(settings.upload_dir, safe_name)
    token = uuid.uuid4().hex[:8]
    temporary = destination.with_name(f".{token}.part")
    backup = destination.with_name(f".{token}.bak")

    temporary.write_bytes(data)
    had_previous = destination.exists()
    if had_previous:
        os.replace(destination, backup)
    os.replace(temporary, destination)

    try:
        with pool.connection() as conn:
            result = ingest_file(destination, conn, embedder)
            row = conn.execute(
                "SELECT id FROM documents WHERE path = %s", (normalise_path(destination),)
            ).fetchone()
    except Exception as error:  # noqa: BLE001 - reported as a failed ingest, never as a crash
        # Put things back as they were before this upload.
        if had_previous:
            os.replace(backup, destination)
        else:
            destination.unlink(missing_ok=True)
        message = str(error)
        friendly = message if isinstance(error, ValueError) and _SAFE_FAILURE_MARKER in message else GENERIC_FAILURE
        return IngestResult(status="failed", chunks=0, document_id=None, filename=safe_name, error=friendly)

    if had_previous:
        backup.unlink(missing_ok=True)
    return IngestResult(
        status=result.status,
        chunks=result.chunks,
        document_id=row[0] if row else None,
        filename=safe_name,
    )


@router.post("/ingest", response_model=IngestResult)
async def ingest(
    request: Request,
    file: UploadFile,
    pool=Depends(get_pool),
    embedder=Depends(get_embedder),
    settings: Settings = Depends(get_settings),
) -> IngestResult:
    """Upload one .pdf or .md file; it is saved under data/uploads and added to the knowledge base."""
    try:
        safe_name = sanitize_filename(file.filename)
        data = await read_limited(file, settings.max_upload_bytes)
    except InvalidUpload as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from None
    finally:
        await file.close()

    try:
        result = await run_in_threadpool(store_and_ingest, data, safe_name, settings, pool, embedder)
    except InvalidUpload as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from None

    logger.info(
        "ingest id=%s file=%s status=%s chunks=%d",
        request_id_of(request), result.filename, result.status, result.chunks,
    )
    return result
