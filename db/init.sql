CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    source_type TEXT NOT NULL,
    path TEXT NOT NULL UNIQUE,
    title TEXT,
    content_hash TEXT NOT NULL,
    doc_date DATE,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    summary TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'
);

CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    section_heading TEXT,
    page_number INTEGER,
    embedding vector(384),
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);

CREATE INDEX chunks_embedding_hnsw_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chunks_tsv_gin_idx ON chunks USING gin (tsv);
CREATE INDEX chunks_document_id_idx ON chunks (document_id);
