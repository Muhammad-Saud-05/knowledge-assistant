import pytest

from src.api.uploads import InvalidUpload, sanitize_filename


def words(label, n=40):
    return " ".join(f"{label}{i}" for i in range(n))


def note(label="alpha"):
    return f"# Note\n\n## Section\n\n{words(label)}\n".encode()


def upload(client, name="notes.md", content=None, content_type="text/markdown"):
    content = note() if content is None else content
    return client.post("/api/ingest", files={"file": (name, content, content_type)})


def count(conn, table):
    return conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def files_under(path):
    return sorted(p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file())


# ------------------------------------------------------------ name sanitising


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("notes.md", "notes.md"),
        ("Report.PDF", "Report.pdf"),
        ("NOTES.MD", "NOTES.md"),
        ("my notes (v2).md", "my notes _v2_.md"),
        ("a.b.c.md", "a.b.c.md"),
        ("naïve café.md", "na_ve caf_.md"),
        ("  padded.md  ", "padded.md"),
        ("trailing.md.", "trailing.md"),
        ("x" * 300 + ".md", "x" * 100 + ".md"),
    ],
)
def test_sanitize_accepts_and_cleans_ordinary_names(raw, expected):
    assert sanitize_filename(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "../../evil.md",
        "..\\evil.md",
        "..\\..\\windows\\evil.md",
        "/etc/passwd.md",
        "C:\\Users\\x\\evil.md",
        "folder/evil.md",
        "folder\\evil.md",
        "\uff0e\uff0e\uff0fevil.md",  # full-width "../" becomes "../" after normalising
        ".hidden.md",
        "..md",
        "CON.md",
        "con.MD",
        "NUL.pdf",
        "COM1.md",
        "lpt9.pdf",
        "CON.backup.md",
        "bad\x00name.md",
        "tab\tname.md",
        "",
        "   ",
        ".md",
        None,
    ],
)
def test_sanitize_refuses_dangerous_names(raw):
    with pytest.raises(InvalidUpload):
        sanitize_filename(raw)


@pytest.mark.parametrize("raw", ["notes.txt", "run.exe", "noextension", "evil.md.exe", "x.md:stream", "page.html", "a.markdown"])
def test_sanitize_refuses_other_file_types_with_415(raw):
    with pytest.raises(InvalidUpload) as error:
        sanitize_filename(raw)
    assert error.value.status_code == 415


# ------------------------------------------------------------------ ingesting


def test_new_file_is_ingested_and_saved_in_the_upload_folder(client, conn, api_settings):
    response = upload(client)

    assert response.status_code == 200
    body = response.json()
    assert (body["status"], body["filename"], body["error"]) == ("new", "notes.md", None)
    assert body["chunks"] >= 1 and body["document_id"] > 0
    assert files_under(api_settings.upload_dir) == ["notes.md"]
    assert (api_settings.upload_dir / "notes.md").read_bytes() == note()
    assert count(conn, "documents") == 1 and count(conn, "chunks") == body["chunks"]


def test_same_file_again_is_unchanged_and_not_duplicated(client, conn):
    first = upload(client).json()
    second = upload(client).json()

    assert second["status"] == "unchanged"
    assert second["document_id"] == first["document_id"]
    assert count(conn, "documents") == 1 and count(conn, "chunks") == first["chunks"]


def test_changed_content_with_the_same_name_is_an_update(client, conn, api_settings):
    upload(client, content=note("old"))
    response = upload(client, content=note("brand"))

    assert response.json()["status"] == "updated"
    assert count(conn, "documents") == 1
    assert "brand0" in conn.execute("SELECT string_agg(text, ' ') FROM chunks").fetchone()[0]
    assert (api_settings.upload_dir / "notes.md").read_bytes() == note("brand")
    assert files_under(api_settings.upload_dir) == ["notes.md"]  # no leftover temporary or backup files


def test_a_pdf_upload_is_accepted(client, api_settings):
    import pymupdf

    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 100), words("pdfword", 60).replace(" ", " "), fontsize=11)
    data = pdf.tobytes()
    pdf.close()

    response = upload(client, "paper.pdf", data, "application/pdf")
    assert response.status_code == 200 and response.json()["status"] in ("new", "failed")
    assert response.json()["filename"] == "paper.pdf"


def test_odd_characters_in_a_name_are_replaced(client, api_settings):
    response = upload(client, "my notes (v2).md")
    assert response.json()["filename"] == "my notes _v2_.md"
    assert files_under(api_settings.upload_dir) == ["my notes _v2_.md"]


@pytest.mark.parametrize("name", ["notes.txt", "run.exe", "noextension", "evil.md.exe", "page.html"])
def test_wrong_extension_is_rejected(client, conn, api_settings, name):
    response = upload(client, name)
    assert response.status_code == 415
    assert count(conn, "documents") == 0
    assert not api_settings.upload_dir.exists() or files_under(api_settings.upload_dir) == []


def test_empty_file_is_rejected(client, conn):
    response = upload(client, content=b"")
    assert response.status_code == 400 and "empty" in response.json()["detail"].lower()
    assert count(conn, "documents") == 0


def test_oversized_file_is_rejected_while_reading(client, conn, api_settings):
    too_big = b"x" * (api_settings.max_upload_bytes + 1)
    response = upload(client, content=too_big)
    assert response.status_code == 413
    assert count(conn, "documents") == 0
    assert not api_settings.upload_dir.exists() or files_under(api_settings.upload_dir) == []


def test_file_at_exactly_the_limit_is_not_rejected_for_size(client, api_settings):
    body = (b"word " * 100000)[: api_settings.max_upload_bytes]
    assert upload(client, content=body).status_code == 200


def test_request_declared_far_too_large_is_refused_before_reading(client, api_settings):
    huge = b"x" * (api_settings.max_upload_bytes + 2 * 1024 * 1024)
    assert upload(client, content=huge).status_code == 413


@pytest.mark.parametrize(
    "name", ["../../evil.md", "..\\evil.md", "..\\..\\evil.md", "/abs/evil.md", "CON.md", ".hidden.md", "con.MD", "sub/dir.md"]
)
def test_dangerous_names_are_rejected_and_nothing_is_written_anywhere(client, conn, tmp_path, api_settings, name):
    response = upload(client, name)

    assert response.status_code in (400, 415)
    assert count(conn, "documents") == 0
    everything = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert everything == [], f"unexpected files written: {everything}"


def test_error_messages_do_not_reveal_server_paths_or_content(client, tmp_path):
    secret = "zebra-secret-content"
    for name, content in (("evil.md.exe", note(secret)), ("../x.md", note(secret)), ("tiny.md", secret.encode())):
        text = upload(client, name, content).text
        assert str(tmp_path) not in text and secret not in text
        assert "data/uploads" not in text and "data\\\\uploads" not in text and ".part" not in text


def test_a_file_too_short_to_index_is_a_failed_result_not_a_crash(client, conn, api_settings):
    response = upload(client, "tiny.md", b"# Tiny\n\nToo short to index.\n")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed" and body["document_id"] is None and body["chunks"] == 0
    assert "20 words" in body["error"]
    assert count(conn, "documents") == 0
    assert files_under(api_settings.upload_dir) == []  # the unusable file was not kept


def test_a_corrupt_pdf_fails_with_a_generic_message(client, conn, api_settings):
    response = upload(client, "broken.pdf", b"this is not really a pdf", "application/pdf")

    body = response.json()
    assert body["status"] == "failed" and body["error"]
    assert str(api_settings.upload_dir) not in response.text and "broken" not in body["error"]
    assert count(conn, "documents") == 0 and files_under(api_settings.upload_dir) == []


def test_a_failed_update_keeps_the_previous_good_version(client, conn, api_settings):
    good = upload(client, content=note("good")).json()
    failed = upload(client, content=b"# Tiny\n\nToo short.\n").json()

    assert failed["status"] == "failed"
    assert (api_settings.upload_dir / "notes.md").read_bytes() == note("good")  # old file restored
    assert files_under(api_settings.upload_dir) == ["notes.md"]
    assert count(conn, "documents") == 1
    assert conn.execute("SELECT id FROM documents").fetchone()[0] == good["document_id"]


def test_missing_file_field_is_a_validation_error(client):
    assert client.post("/api/ingest").status_code == 422


# ------------------------------------------------------------------ documents


def test_documents_list_starts_empty(client):
    assert client.get("/api/documents").json() == []


def test_documents_list_shows_details_and_chunk_counts_newest_first(client, conn):
    first = upload(client, "first.md", note("one")).json()
    second = upload(client, "second.md", note("two")).json()

    documents = client.get("/api/documents").json()

    assert [d["id"] for d in documents] == [second["document_id"], first["document_id"]]
    newest = documents[0]
    assert set(newest) == {"id", "title", "path", "source_type", "doc_date", "ingested_at", "chunks"}
    assert newest["title"] == "Note" and newest["source_type"] == "markdown"
    assert newest["path"].endswith("second.md") and newest["chunks"] == second["chunks"]
    assert newest["doc_date"] is not None and newest["ingested_at"]


# --------------------------------------------------------------------- delete


def test_delete_removes_the_document_its_chunks_and_the_uploaded_file(client, conn, api_settings):
    document_id = upload(client).json()["document_id"]
    assert (api_settings.upload_dir / "notes.md").exists()

    response = client.delete(f"/api/documents/{document_id}")

    assert response.status_code == 200
    assert response.json() == {"deleted": True, "file_removed": True}
    assert count(conn, "documents") == 0 and count(conn, "chunks") == 0
    assert not (api_settings.upload_dir / "notes.md").exists()
    assert client.get("/api/documents").json() == []


def test_delete_never_removes_a_file_outside_the_upload_folder(client, conn, tmp_path, add_document):
    outside = tmp_path / "sample_corpus" / "precious.md"
    outside.parent.mkdir()
    outside.write_text("do not delete me")
    document_id, _ = add_document(conn, "Precious", str(outside), [("some text here", None, None)])

    response = client.delete(f"/api/documents/{document_id}")

    assert response.status_code == 200 and response.json() == {"deleted": True, "file_removed": False}
    assert outside.read_text() == "do not delete me"
    assert count(conn, "documents") == 0


def test_delete_with_a_relative_path_to_a_corpus_file_leaves_it_alone(client, conn, tmp_path, monkeypatch, add_document):
    monkeypatch.chdir(tmp_path)
    corpus = tmp_path / "sample_corpus"
    corpus.mkdir()
    (corpus / "keep.md").write_text("keep")
    document_id, _ = add_document(conn, "Keep", "sample_corpus/keep.md", [("some text here", None, None)])

    assert client.delete(f"/api/documents/{document_id}").json()["file_removed"] is False
    assert (corpus / "keep.md").read_text() == "keep"


def test_delete_ignores_a_stored_path_that_climbs_out_of_the_upload_folder(client, conn, tmp_path, api_settings, add_document):
    api_settings.upload_dir.mkdir(parents=True)
    victim = tmp_path / "victim.md"
    victim.write_text("keep")
    tricky = str(api_settings.upload_dir / ".." / ".." / "victim.md")
    document_id, _ = add_document(conn, "Tricky", tricky, [("some text here", None, None)])

    assert client.delete(f"/api/documents/{document_id}").json()["file_removed"] is False
    assert victim.read_text() == "keep"


def test_delete_unknown_id_is_404(client):
    response = client.delete("/api/documents/99999")
    assert response.status_code == 404


def test_delete_with_a_non_numeric_id_is_a_validation_error(client):
    assert client.delete("/api/documents/abc").status_code == 422
