# Project Brief: Personal Knowledge Assistant

## Problem

Personal notes, PDFs, and learning materials are hard to search and recall. Keyword search over scattered files misses relevant material, and finding where something was written takes too long.

## Goals

- Provide conversational access to my own documents, with answers that cite their sources.
- Run fully locally, so no document content leaves my machine (privacy).

## Non-goals

- Live Gmail sync
- Multi-user accounts
- Cloud hosting
- Mobile app
- Fine-tuning models

## User stories

1. As a student, I want to ask a question in plain language and get an answer drawn from my notes, so I can recall what I learned without rereading everything.
2. As a user, I want every answer to cite the document (and page or section) it came from, so I can verify it and read the original.
3. As a user, I want to add new notes and PDFs and have them become searchable, so my knowledge base grows with my materials.
4. As a privacy-conscious user, I want all processing to happen on my own machine, so my personal documents are never sent to a third party.
5. As a user, I want the assistant to say when my documents do not contain an answer, so I am not misled by made-up content.

## Tech stack

| Layer | Choice |
| --- | --- |
| Language | Python |
| Database | PostgreSQL + pgvector |
| Embeddings | sentence-transformers (bge-small-en-v1.5) |
| Reranking | Cross-encoder reranker |
| LLM | Ollama with llama3.2:3b |
| API | FastAPI |
| Frontend | React + Vite + TypeScript |
