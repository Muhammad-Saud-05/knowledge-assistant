from pathlib import Path

import pytest

from src.ingestion.pipeline import (
    delete_document,
    find_documents,
    get_document_info,
    ingest_file,
    ingest_files,
    normalise_path,
)


def long_text(label: str, sentences: int = 6) -> str:
    """A paragraph of unique text well above the 20-word minimum."""
    return " ".join(f"{label} sentence {i} says something quite different." for i in range(sentences))


def write_note(path: Path, **sections: str) -> Path:
    body = "# Title\n\n" + "".join(f"## {name}\n\n{text}\n\n" for name, text in sections.items())
    path.write_text(body, encoding="utf-8")
    return path


def count(conn, table: str) -> int:
    return conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ------------------------------------------------------------- ingest basics


def test_new_file_is_ingested_with_all_fields(conn, embedder, workdir):
    note = write_note(workdir / "notes.md", Alpha=long_text("alpha"), Beta=long_text("beta"))
    result = ingest_file(note, conn, embedder)

    assert (result.status, result.chunks) == ("new", 2)
    doc = conn.execute(
        "SELECT source_type, path, title, content_hash, doc_date FROM documents"
    ).fetchone()
    assert doc[0] == "markdown" and doc[1] == "notes.md" and doc[2] == "Title"
    assert len(doc[3]) == 64 and doc[4] is not None
    rows = conn.execute(
        "SELECT chunk_index, section_heading, page_number, vector_dims(embedding) "
        "FROM chunks ORDER BY chunk_index"
    ).fetchall()
    assert rows == [(0, "Title > Alpha", None, 384), (1, "Title > Beta", None, 384)]


def test_same_file_twice_is_unchanged_and_not_duplicated(conn, embedder, workdir):
    note = write_note(workdir / "notes.md", Alpha=long_text("alpha"))
    ingest_file(note, conn, embedder)
    embedded_before = embedder.texts_embedded

    again = ingest_file(note, conn, embedder)

    assert (again.status, again.chunks) == ("unchanged", 1)
    assert embedder.texts_embedded == embedded_before  # nothing was re-embedded
    assert count(conn, "documents") == 1 and count(conn, "chunks") == 1


def test_changed_file_replaces_old_document_and_chunks(conn, embedder, workdir):
    note = write_note(workdir / "notes.md", Alpha=long_text("old"), Beta=long_text("old2"))
    ingest_file(note, conn, embedder)
    old_hash = conn.execute("SELECT content_hash FROM documents").fetchone()[0]

    write_note(note, Gamma=long_text("brand new"))
    result = ingest_file(note, conn, embedder)

    assert (result.status, result.chunks) == ("updated", 1)
    assert count(conn, "documents") == 1 and count(conn, "chunks") == 1
    text = conn.execute("SELECT text FROM chunks").fetchone()[0]
    assert "brand new" in text and "old" not in text
    assert conn.execute("SELECT content_hash FROM documents").fetchone()[0] != old_hash


def test_chunks_under_minimum_words_are_dropped(conn, embedder, workdir):
    note = write_note(workdir / "notes.md", Tiny="Just five short words.", Real=long_text("real"))
    result = ingest_file(note, conn, embedder, min_chunk_words=20)
    assert result.chunks == 1
    assert "short words" not in conn.execute("SELECT string_agg(text, ' ') FROM chunks").fetchone()[0]

    other = write_note(workdir / "other.md", Tiny="Just five short words.", Real=long_text("real"))
    assert ingest_file(other, conn, embedder, min_chunk_words=3).chunks == 2


def test_chunk_index_is_consecutive_after_dropping(conn, embedder, workdir):
    note = write_note(
        workdir / "notes.md",
        A="Tiny first section.",
        B=long_text("b"),
        C="Tiny middle section.",
        D=long_text("d"),
        E=long_text("e"),
    )
    ingest_file(note, conn, embedder)
    indexes = [r[0] for r in conn.execute("SELECT chunk_index FROM chunks ORDER BY chunk_index")]
    assert indexes == [0, 1, 2]


def test_file_with_no_usable_chunks_fails_without_writing(conn, embedder, workdir):
    note = write_note(workdir / "notes.md", Tiny="Far too short.")
    with pytest.raises(ValueError, match="at least 20 words"):
        ingest_file(note, conn, embedder)
    assert count(conn, "documents") == 0


def test_dry_run_reports_but_writes_nothing(conn, embedder, workdir):
    note = write_note(workdir / "notes.md", Alpha=long_text("alpha"))
    result = ingest_file(note, conn, embedder, dry_run=True)
    assert (result.status, result.chunks) == ("new", 1)
    assert count(conn, "documents") == 0 and embedder.texts_embedded == 0

    ingest_file(note, conn, embedder)
    assert ingest_file(note, conn, embedder, dry_run=True).status == "unchanged"
    write_note(note, Beta=long_text("beta"))
    assert ingest_file(note, conn, embedder, dry_run=True).status == "updated"
    assert count(conn, "chunks") == 1  # still the original version


# ------------------------------------------------------------------- delete


def test_deleting_a_document_removes_its_chunks(conn, embedder, workdir):
    first = write_note(workdir / "a.md", Alpha=long_text("a1"), Beta=long_text("a2"))
    second = write_note(workdir / "b.md", Alpha=long_text("b1"))
    ingest_file(first, conn, embedder)
    ingest_file(second, conn, embedder)
    assert get_document_info("a.md", conn) == ("Title", 2)

    assert delete_document("a.md", conn) is True

    assert count(conn, "documents") == 1 and count(conn, "chunks") == 1
    assert get_document_info("a.md", conn) is None
    assert delete_document("a.md", conn) is False  # nothing left to delete


# ---------------------------------------------------------- failures & paths


def test_one_failing_file_does_not_stop_the_batch(conn, embedder, workdir):
    good1 = write_note(workdir / "good1.md", Alpha=long_text("g1"))
    bad = workdir / "broken.pdf"
    bad.write_bytes(b"this is not a pdf")
    good2 = write_note(workdir / "good2.md", Alpha=long_text("g2"))

    results = list(ingest_files([good1, bad, good2], conn, embedder))

    assert [r.status for r in results] == ["new", "failed", "new"]
    assert results[1].error and "broken.pdf" in results[1].path
    paths = [r[0] for r in conn.execute("SELECT path FROM documents ORDER BY path")]
    assert paths == ["good1.md", "good2.md"]


def test_failed_update_rolls_back_and_keeps_the_old_version(conn, embedder, wrong_size_embedder, workdir):
    note = write_note(workdir / "notes.md", Alpha=long_text("original"))
    ingest_file(note, conn, embedder)

    write_note(note, Beta=long_text("replacement"))
    results = list(ingest_files([note], conn, wrong_size_embedder))

    assert results[0].status == "failed"
    assert count(conn, "documents") == 1 and count(conn, "chunks") == 1
    assert "original" in conn.execute("SELECT text FROM chunks").fetchone()[0]
    # The connection is still usable, and the new version can be ingested afterwards.
    assert ingest_file(note, conn, embedder).status == "updated"


def test_failed_new_file_leaves_nothing_behind(conn, wrong_size_embedder, workdir):
    note = write_note(workdir / "notes.md", Alpha=long_text("alpha"))
    results = list(ingest_files([note], conn, wrong_size_embedder))
    assert results[0].status == "failed"
    assert count(conn, "documents") == 0 and count(conn, "chunks") == 0


def test_path_is_stored_relative_with_forward_slashes(conn, embedder, workdir):
    (workdir / "sub" / "deeper").mkdir(parents=True)
    note = write_note(workdir / "sub" / "deeper" / "n.md", Alpha=long_text("alpha"))
    ingest_file(note, conn, embedder)  # absolute path given
    assert conn.execute("SELECT path FROM documents").fetchone()[0] == "sub/deeper/n.md"


def test_path_outside_working_directory_stays_absolute(tmp_path):
    inside, outside = tmp_path / "inside", tmp_path / "outside"
    inside.mkdir()
    outside.mkdir()
    assert normalise_path(outside / "n.md", base=inside) == (outside / "n.md").resolve().as_posix()
    assert normalise_path(inside / "a" / "n.md", base=inside) == "a/n.md"


def test_find_documents_walks_folders_for_supported_types(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    for name in ("x.md", "a/y.pdf", "a/b/z.MD", "a/skip.txt", "a/b/skip.png"):
        (tmp_path / name).write_bytes(b"")
    found = sorted(p.relative_to(tmp_path).as_posix() for p in find_documents(tmp_path))
    assert found == ["a/b/z.MD", "a/y.pdf", "x.md"]
