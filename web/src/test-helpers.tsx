// Shared helpers for the interface tests: a fake back end whose answer streams the test controls.

import { act, fireEvent, screen, waitFor } from '@testing-library/react'
import { expect, vi } from 'vitest'
import type { DoneEvent, Health, SourceItem } from './api/types'

const encoder = new TextEncoder()
export const sse = (event: string, data: unknown) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`

/** A response whose events the test sends one at a time, like a real stream. */
export class Pipe {
  response: Response
  private controller!: ReadableStreamDefaultController<Uint8Array>

  constructor(signal?: AbortSignal | null) {
    const body = new ReadableStream<Uint8Array>({ start: (controller) => (this.controller = controller) })
    signal?.addEventListener('abort', () => {
      try {
        this.controller.error(new DOMException('Aborted', 'AbortError'))
      } catch {
        // already closed
      }
    })
    this.response = new Response(body, { status: 200, headers: { 'Content-Type': 'text/event-stream' } })
  }

  async send(text: string) {
    await act(async () => {
      this.controller.enqueue(encoder.encode(text))
      await new Promise((resolve) => setTimeout(resolve, 0))
    })
  }

  async end() {
    await act(async () => {
      this.controller.close()
      await new Promise((resolve) => setTimeout(resolve, 0))
    })
  }
}

export const HEALTHY: Health = {
  status: 'ok',
  database: { ok: true, documents: 4, chunks: 26, detail: null },
  ollama: { ok: true, model: 'llama3.2:3b', model_installed: true, detail: null },
}

export const SOURCES: SourceItem[] = [
  {
    number: 1, chunk_id: 11, title: 'Rate Limiting Notes', path: 'notes/rate.md', page_number: null,
    section_heading: 'Token bucket', text: 'A bucket holds up to a fixed number of tokens. '.repeat(8),
    vector_rank: 1, keyword_rank: 2, rerank_rank: 1, rerank_score: 6.42,
  },
  {
    number: 2, chunk_id: 12, title: "A Beginner's Guide", path: 'guide.pdf', page_number: 5,
    section_heading: '4. Retrieval', text: 'When a question arrives, the system first retrieves candidates.',
    vector_rank: 2, keyword_rank: null, rerank_rank: 2, rerank_score: 3.1,
  },
  {
    number: 3, chunk_id: 13, title: 'Fixed Window Notes', path: 'notes/fixed.md', page_number: null,
    section_heading: null, text: 'Fixed window counters reset at the end of every window.',
    vector_rank: 3, keyword_rank: 1,
  },
]

export function doneEvent(overrides: Partial<Omit<DoneEvent, 'text'>> = {}, text = 'A token bucket refills steadily [1].'): DoneEvent {
  return {
    text,
    refused: false,
    cited: [1],
    invalid_citations: [],
    uncited: false,
    validation: { cited_valid: [1], cited_invalid: [], has_no_citations: false, is_not_found: false },
    timings: { retrieval: 0.1, first_token: 0.6, total: 3.2, queue_wait: 0 },
    ...overrides,
  }
}

export interface FakeServer {
  askCalls: { body: { question: string; mode?: string }; signal: AbortSignal | null | undefined }[]
  onAsk: (init: RequestInit) => Response | Promise<Response>
  health: Health
  /** Answer the next question with a stream the test controls. */
  nextStream: () => { pipe: () => Pipe }
}

/** Replace fetch with a fake back end (health, documents, ask). Call from beforeEach. */
export function installFakeServer(): FakeServer {
  const server: FakeServer = {
    askCalls: [],
    health: HEALTHY,
    onAsk: () => {
      throw new Error('this test did not expect a question')
    },
    nextStream() {
      let current: Pipe | undefined
      server.onAsk = (init) => {
        current = new Pipe(init.signal)
        return current.response
      }
      return {
        pipe: () => {
          if (!current) throw new Error('no question was asked yet')
          return current
        },
      }
    },
  }
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url === '/api/health') return Response.json(server.health)
      if (url === '/api/documents') return Response.json([])
      if (url === '/api/ask') {
        server.askCalls.push({ body: JSON.parse(String(init?.body)), signal: init?.signal })
        return server.onAsk(init ?? {})
      }
      throw new Error(`unexpected request to ${url}`)
    }),
  )
  return server
}

export const questionBox = () => screen.getByLabelText('Your question') as HTMLTextAreaElement

export async function askQuestion(server: FakeServer, question: string) {
  const before = server.askCalls.length
  fireEvent.change(questionBox(), { target: { value: question } })
  fireEvent.keyDown(questionBox(), { key: 'Enter' })
  await waitFor(() => expect(server.askCalls.length).toBe(before + 1))
}

/** Ask a question and finish its answer in one go. */
export async function askAndFinish(server: FakeServer, question: string, answer = 'An answer.', done: Partial<Omit<DoneEvent, 'text'>> = {}) {
  const stream = server.nextStream()
  await askQuestion(server, question)
  await stream.pipe().send(sse('sources', SOURCES) + sse('token', { text: answer }) + sse('done', doneEvent(done, answer)))
  await stream.pipe().end()
}
