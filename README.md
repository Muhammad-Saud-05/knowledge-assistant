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

Documents (`.pdf`, `.md`) are parsed into sections and cut into chunks that keep their heading and page number for citations, then embedded and stored in the database. Searching and answering are not built yet.

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

### Store documents in the database

With the database running (`docker compose up -d`) and your `.env` filled in, embed and store a file or a whole folder (searched recursively for `.pdf` and `.md`):

```powershell
python -m src.ingestion.ingest docs\kb-samples
```

It prints one line per file (`new`, `updated`, `unchanged` or `failed`) with the chunk count and time, then totals. Running it again is safe: a file whose path and content hash are unchanged is skipped, and a changed file replaces its old chunks. Each file is stored in a single transaction, and one failing file does not stop the others.

Useful options:

```powershell
python -m src.ingestion.ingest docs\kb-samples --dry-run      # show what would happen, write nothing
python -m src.ingestion.ingest docs\kb-samples --min-words 30 # drop chunks shorter than 30 words (default 20)
python -m src.ingestion.ingest --delete docs/kb-samples/study-habits-notes.md  # remove one document and its chunks
```

Paths are stored relative to the folder you run the command from, so use the same form with `--delete`.

### Try a search (sanity check only)

```powershell
python scripts\smoke_search.py "how does a token bucket work"
```

This embeds the question and prints the 5 nearest chunks with their cosine similarity. It only checks that storage works; real retrieval comes later.

### The embedding model

Embeddings use `BAAI/bge-small-en-v1.5` on the CPU. The model (about 130 MB) downloads once from Hugging Face the first time it is used, and is cached afterwards. After that it can run fully offline:

```powershell
$env:HF_HUB_OFFLINE = "1"
```

### Tests

```powershell
python -m pytest                 # everything
python -m pytest -m "not slow"   # skip the test that loads the real embedding model
```

The database tests create a temporary `kb_test` database on the same server and drop it afterwards, so your real data is never touched. If Postgres is not running they are skipped.
