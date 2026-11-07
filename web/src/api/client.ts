// Typed functions for the back end. All paths are relative (/api/...): the Vite dev server
// forwards them to the back end, so the browser sees a single origin.

import { dispatchSseEvent } from './events'
import { SseParser } from './sse'
import type { AskHandlers, AskRequest, DeleteResult, DocumentItem, Health, IngestResult } from './types'

export const NETWORK_MESSAGE =
  "I couldn't reach the server. Is it running? Start it with: uvicorn src.api.main:app --host 127.0.0.1 --port 8000"
export const INTERRUPTED_MESSAGE = 'The connection to the server was interrupted before the answer finished.'

/** An error whose message is fit to show to the user. */
export class ApiError extends Error {
  status: number | null

  constructor(message: string, status: number | null = null) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

function isAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError'
}

/** A readable message for a failed HTTP response. */
async function errorFromResponse(response: Response): Promise<ApiError> {
  let detail = ''
  try {
    const body: unknown = await response.json()
    if (typeof body === 'object' && body !== null && 'detail' in body && typeof body.detail === 'string') {
      detail = body.detail
    }
  } catch {
    // the body was not JSON: fall back to the generic messages below
  }
  const status = response.status
  if (detail && status < 500) return new ApiError(detail, status)
  if (status === 403) return new ApiError('The server refused this request.', status)
  if (status === 413) return new ApiError('That is too large for the server to accept.', status)
  if (status === 422) return new ApiError('The server could not accept that request. Check the question and options.', status)
  if (status >= 500) return new ApiError('The server ran into a problem. Please try again.', status)
  return new ApiError(`The server answered with an error (${status}).`, status)
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(path, init)
  } catch (error) {
    if (isAbort(error)) throw error
    throw new ApiError(NETWORK_MESSAGE)
  }
  if (!response.ok) throw await errorFromResponse(response)
  return (await response.json()) as T
}

export function getHealth(signal?: AbortSignal): Promise<Health> {
  return requestJson<Health>('/api/health', { signal })
}

export function getDocuments(signal?: AbortSignal): Promise<DocumentItem[]> {
  return requestJson<DocumentItem[]>('/api/documents', { signal })
}

export function ingestFile(file: File, signal?: AbortSignal): Promise<IngestResult> {
  const form = new FormData()
  form.append('file', file) // no Content-Type header: the browser adds it, with the boundary
  return requestJson<IngestResult>('/api/ingest', { method: 'POST', body: form, signal })
}

export function deleteDocument(id: number, signal?: AbortSignal): Promise<DeleteResult> {
  return requestJson<DeleteResult>(`/api/documents/${id}`, { method: 'DELETE', signal })
}

/**
 * Ask a question and report the streamed answer through `handlers`.
 *
 * Resolves when the stream ends normally or when `signal` aborts it (the caller already
 * knows it asked for that). Throws an ApiError, with a message fit to show, if the request
 * is refused or the connection fails. Errors the server reports inside the stream arrive
 * through handlers.onError.
 */
export async function askQuestion(request: AskRequest, handlers: AskHandlers, signal?: AbortSignal): Promise<void> {
  let response: Response
  try {
    response = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
      body: JSON.stringify(request),
      signal,
    })
  } catch (error) {
    if (isAbort(error)) return
    throw new ApiError(NETWORK_MESSAGE)
  }
  if (!response.ok) throw await errorFromResponse(response)
  if (!response.body) throw new ApiError(INTERRUPTED_MESSAGE)

  const reader = response.body.getReader()
  const decoder = new TextDecoder() // streaming mode below: a character split across chunks stays whole
  const parser = new SseParser()
  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      for (const event of parser.feed(decoder.decode(value, { stream: true }))) {
        dispatchSseEvent(event, handlers)
      }
    }
    for (const event of parser.feed(decoder.decode())) dispatchSseEvent(event, handlers)
    for (const event of parser.finish()) dispatchSseEvent(event, handlers)
  } catch (error) {
    if (isAbort(error)) return
    throw new ApiError(INTERRUPTED_MESSAGE)
  } finally {
    reader.cancel().catch(() => {}) // release the connection (a no-op if the stream already ended)
  }
}
