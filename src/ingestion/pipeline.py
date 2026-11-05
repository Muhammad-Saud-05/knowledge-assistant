"""Parse, chunk, embed and store documents in the database.

Each file is handled in a single transaction, so a failure can never leave a
half-ingested or half-deleted document behind. ingest_file() and
delete_document() commit when they finish; if the connection already has an
open transaction, it is committed together with them.
"""

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path

from pgvector import Vector

from .chunker import chunk_document
from .models import Chunk
from .parsers import MARKDOWN_EXTENSIONS, PDF_EXTENSIONS, parse_file

DEFAULT_MIN_CHUNK_WORDS = 20
SUPPORTED_EXTENSIONS = MARKDOWN_EXTENSIONS | PDF_EXTENSIONS


@dataclass
class IngestResult:
    path: str
    status: str  # "new", "updated", "unchanged" or "failed"
    chunks: int = 0
    seconds: float = 0.0
    error: str | None = None


def normalise_path(path: str | Path, base: str | Path | None = None) -> str:
    """Path relative to `base` (default: current directory) with forward slashes.

    Files outside `base` keep their absolute path.
    """
    resolved = Path(path).resolve()
    root = Path(base).resolve() if base else Path.cwd().resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()


def find_documents(target: str | Path) -> list[Path]:
    """The file itself, or every supported file under a folder (recursively)."""
    target = Path(target)
    if target.is_file():
        return [target]
    return sorted(
        p for p in target.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _clean(text: str) -> str:
    return text.replace("\x00", "")  # Postgres text cannot hold NUL characters


def _usable_chunks(chunks: list[Chunk], min_chunk_words: int) -> list[Chunk]:
    """Drop tiny chunks (cover pages, one-line intros) and renumber the rest 0..n-1."""
    kept = [c for c in chunks if len(c.text.split()) >= min_chunk_words]
    return [replace(c, chunk_index=i) for i, c in enumerate(kept)]


def ingest_file(
    path: str | Path,
    conn,
    embedder,
    min_chunk_words: int = DEFAULT_MIN_CHUNK_WORDS,
    dry_run: bool = False,
) -> IngestResult:
    """Store one file: "new", "updated" (content changed) or "unchanged" (skipped).

    With dry_run=True nothing is embedded or written; the result says what
    would happen. Raises on failure, after rolling back this file's changes.
    """
    started = time.perf_counter()
    stored_path = normalise_path(path)

    doc = parse_file(path)
    chunks = _usable_chunks(chunk_document(doc), min_chunk_words)

    with conn.transaction():
        existing = conn.execute(
            "SELECT id, content_hash FROM documents WHERE path = %s", (stored_path,)
        ).fetchone()

        if existing and existing[1] == doc.content_hash:
            stored_chunks = conn.execute(
                "SELECT count(*) FROM chunks WHERE document_id = %s", (existing[0],)
            ).fetchone()[0]
            result = IngestResult(stored_path, "unchanged", stored_chunks)
        elif not chunks:
            raise ValueError(
                f"no chunk has at least {min_chunk_words} words "
                "(empty or scanned PDF? try a lower --min-words)"
            )
        elif dry_run:
            result = IngestResult(stored_path, "updated" if existing else "new", len(chunks))
        else:
            vectors = embedder.embed_documents([c.text for c in chunks])
            if len(vectors) != len(chunks):
                raise RuntimeError("embedder returned a different number of vectors than chunks")

            if existing:
                # ON DELETE CASCADE removes the old chunks and their vectors.
                conn.execute("DELETE FROM documents WHERE id = %s", (existing[0],))
            document_id = conn.execute(
                """
                INSERT INTO documents (source_type, path, title, content_hash, doc_date)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (doc.source_type, stored_path, _clean(doc.title), doc.content_hash, doc.doc_date),
            ).fetchone()[0]

            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO chunks
                        (document_id, chunk_index, text, section_heading, page_number, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (
                            document_id,
                            c.chunk_index,
                            _clean(c.text),
                            _clean(c.section_heading) if c.section_heading else None,
                            c.page_number,
                            Vector(vector),
                        )
                        for c, vector in zip(chunks, vectors)
                    ],
                )
            result = IngestResult(stored_path, "updated" if existing else "new", len(chunks))

    conn.commit()
    result.seconds = time.perf_counter() - started
    return result


def ingest_files(
    paths: Iterable[str | Path],
    conn,
    embedder,
    min_chunk_words: int = DEFAULT_MIN_CHUNK_WORDS,
    dry_run: bool = False,
) -> Iterator[IngestResult]:
    """Ingest many files one by one; a failure is reported and the rest carry on."""
    for path in paths:
        started = time.perf_counter()
        try:
            yield ingest_file(path, conn, embedder, min_chunk_words, dry_run)
        except Exception as error:
            conn.rollback()
            message = (str(error).strip().splitlines() or [type(error).__name__])[0][:200]
            yield IngestResult(
                normalise_path(path),
                "failed",
                seconds=time.perf_counter() - started,
                error=f"{type(error).__name__}: {message}",
            )


def get_document_info(path: str | Path, conn) -> tuple[str, int] | None:
    """(title, chunk count) of a stored document, or None if it is not stored."""
    row = conn.execute(
        """
        SELECT d.title, count(c.id)
        FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
        WHERE d.path = %s
        GROUP BY d.id
        """,
        (normalise_path(path),),
    ).fetchone()
    conn.rollback()  # end the read-only transaction
    return (row[0], row[1]) if row else None


def delete_document(path: str | Path, conn) -> bool:
    """Delete a document by path (its chunks and vectors go with it via ON DELETE CASCADE)."""
    with conn.transaction():
        deleted = conn.execute(
            "DELETE FROM documents WHERE path = %s RETURNING id", (normalise_path(path),)
        ).fetchone()
    conn.commit()
    return deleted is not None
