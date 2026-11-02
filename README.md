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
