# Knowledge Assistant

A personal AI knowledge assistant that lets me chat with my own notes, PDFs, and learning materials. It uses retrieval-augmented generation (RAG): relevant passages are found in my documents and a local language model answers using them, with citations. Everything runs locally for privacy.

## Quick start

1. Copy the example environment file:

   ```powershell
   Copy-Item .env.example .env
   ```

2. Open `.env` and replace `change-me` with a strong `POSTGRES_PASSWORD`.
3. Start the database:

   ```powershell
   docker compose up -d
   ```

The database listens on `127.0.0.1:5433` (localhost only).

See [docs/project-brief.md](docs/project-brief.md) for goals and scope.

## Ingestion (work in progress)

Documents (`.pdf`, `.md`) are parsed into sections and cut into chunks that keep their heading and page number for citations. Nothing is written to the database or embedded yet.

Set up Python once:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Inspect how a file or a folder of files is parsed and chunked:

```powershell
python -m src.ingestion.inspect_cli path\to\file-or-folder
```

For each document it prints the title, type, date, number of chunks, average chunk size and the first three chunks. It warns about empty or near-empty PDF pages, which usually means a cover page or a scanned PDF with no extractable text.

Run the tests:

```powershell
python -m pytest
```
