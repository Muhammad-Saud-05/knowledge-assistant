import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import { NETWORK_MESSAGE } from './api/client'
import type { DoneEvent, Health, SourceItem } from './api/types'

// ---------------------------------------------------------------- a fake server

const encoder = new TextEncoder()
const sse = (event: string, data: unknown) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`

/** A response whose events the test sends one at a time, like a real stream. */
class Pipe {
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

const HEALTHY: Health = {
  status: 'ok',
  database: { ok: true, documents: 4, chunks: 26, detail: null },
  ollama: { ok: true, model: 'llama3.2:3b', model_installed: true, detail: null },
}

const SOURCES: SourceItem[] = [
  {
    number: 1, chunk_id: 11, title: 'Rate Limiting Notes', path: 'notes/rate.md', page_number: null,
    section_heading: 'Token bucket', text: 'A bucket holds up to a fixed number of tokens. '.repeat(8),
    vector_rank: 1, keyword_rank: 2,
  },
  {
    number: 2, chunk_id: 12, title: "A Beginner's Guide", path: 'guide.pdf', page_number: 5,
    section_heading: '4. Retrieval', text: 'When a question arrives, the system first retrieves candidates.',
    vector_rank: 2, keyword_rank: null,
  },
]

function doneEvent(overrides: Partial<Omit<DoneEvent, 'text'>> = {}, text = 'A token bucket refills steadily [1].'): DoneEvent {
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

let health: Health
let askCalls: { body: { question: string }; signal: AbortSignal | null | undefined }[]
let onAsk: (init: RequestInit) => Response | Promise<Response>

beforeEach(() => {
  health = HEALTHY
  askCalls = []
  onAsk = () => {
    throw new Error('this test did not expect a question')
  }
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url === '/api/health') return Response.json(health)
      if (url === '/api/ask') {
        askCalls.push({ body: JSON.parse(String(init?.body)), signal: init?.signal })
        return onAsk(init ?? {})
      }
      throw new Error(`unexpected request to ${url}`)
    }),
  )
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

/** Make the fake server answer the next question with a stream the test controls. */
function nextStream(): { pipe: () => Pipe } {
  let current: Pipe | undefined
  onAsk = (init) => {
    current = new Pipe(init.signal)
    return current.response
  }
  return {
    pipe: () => {
      if (!current) throw new Error('no question was asked yet')
      return current
    },
  }
}

function box() {
  return screen.getByLabelText('Your question')
}

function type(question: string) {
  fireEvent.change(box(), { target: { value: question } })
}

function pressEnter(init: KeyboardEventInit = {}) {
  fireEvent.keyDown(box(), { key: 'Enter', ...init })
}

async function ask(question: string) {
  type(question)
  pressEnter()
  await waitFor(() => expect(askCalls.length).toBeGreaterThan(0))
}

// ------------------------------------------------------------------------ tests

describe('first visit', () => {
  it('shows a welcome message instead of an empty page', () => {
    render(<App />)
    expect(screen.getByRole('heading', { name: 'Ask about your documents' })).toBeTruthy()
  })

  it('shows a green status when the database and Ollama work', async () => {
    render(<App />)
    const status = await screen.findByText('Ready - 4 documents, 26 passages')
    expect(status.closest('.status')?.className).toContain('status-ok')
  })

  it('explains what is wrong when Ollama is not running', async () => {
    health = { ...HEALTHY, status: 'degraded', ollama: { ok: false, model: 'llama3.2:3b', model_installed: null, detail: 'x' } }
    render(<App />)
    const status = await screen.findByText("Ollama isn't running")
    expect(status.closest('.status')?.className).toContain('status-problem')
  })

  it('explains when the server itself cannot be reached', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => Promise.reject(new TypeError('Failed to fetch'))))
    render(<App />)
    expect(await screen.findByText("Can't reach the server")).toBeTruthy()
  })
})

describe('sending a question', () => {
  it('sends with Enter, clears the box and shows the question', async () => {
    const server = nextStream()
    render(<App />)
    await ask('How does a token bucket work?')

    expect(askCalls[0].body).toEqual({ question: 'How does a token bucket work?' })
    expect((box() as HTMLTextAreaElement).value).toBe('')
    expect(screen.getByText('How does a token bucket work?')).toBeTruthy()
    expect(screen.queryByRole('heading', { name: 'Ask about your documents' })).toBeNull()
    await server.pipe().end()
  })

  it('does not send on Shift+Enter, on an empty box, or while composing text', () => {
    render(<App />)
    type('A line')
    pressEnter({ shiftKey: true })
    pressEnter({ isComposing: true })
    type('   ')
    pressEnter()
    expect(askCalls).toHaveLength(0)
    expect((box() as HTMLTextAreaElement).value).toBe('   ')
  })

  it('has a Send button that is disabled until there is something to send', () => {
    nextStream()
    render(<App />)
    const send = screen.getByRole('button', { name: 'Send' }) as HTMLButtonElement
    expect(send.disabled).toBe(true)
    type('hello')
    expect(send.disabled).toBe(false)
    fireEvent.click(send)
    expect(askCalls).toHaveLength(1)
  })
})

describe('the answer as it arrives', () => {
  it('shows searching, then the sources, then the streaming text, then the finished answer', async () => {
    const server = nextStream()
    render(<App />)
    await ask('How does a token bucket work?')

    expect(await screen.findByText('Searching your documents...')).toBeTruthy()

    await server.pipe().send(sse('sources', SOURCES))
    const sources = screen.getByRole('region', { name: 'Sources' })
    expect(within(sources).getAllByRole('listitem')).toHaveLength(2)
    expect(within(sources).getByText('Rate Limiting Notes')).toBeTruthy()
    expect(within(sources).getByText('Token bucket')).toBeTruthy()
    expect(within(sources).getByText('p. 5 - 4. Retrieval')).toBeTruthy()
    // the sources are already on screen; the status line now says what the wait is for
    expect(screen.queryByText('Searching your documents...')).toBeNull()
    expect(screen.getByText(/Reading your sources - on this CPU the first words can take 30 to 40 seconds/)).toBeTruthy()

    await server.pipe().send(sse('token', { text: 'A token ' }) + sse('token', { text: 'bucket refills' }))
    expect(screen.queryByText(/Reading your sources/)).toBeNull()
    expect(screen.getByText('A token bucket refills')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Stop' })).toBeTruthy()

    await server.pipe().send(sse('done', doneEvent()))
    expect(screen.getByText('A token bucket refills steadily [1].')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Send' })).toBeTruthy()
    expect(screen.getByText(/finished in 3\.2s/)).toBeTruthy()
    await server.pipe().end()
  })

  it('shows only a 150-character preview of each source and marks the cited one', async () => {
    const server = nextStream()
    render(<App />)
    await ask('q')
    await server.pipe().send(sse('sources', SOURCES) + sse('token', { text: 'x' }) + sse('done', doneEvent()))

    const sources = screen.getByRole('region', { name: 'Sources' })
    const previews = sources.querySelectorAll('.source-text')
    expect(previews[0].textContent!.length).toBeLessThanOrEqual(153)
    expect(previews[0].textContent!.endsWith('...')).toBe(true)
    expect(previews[1].textContent).toBe(SOURCES[1].text)
    expect(within(sources).getAllByText('cited')).toHaveLength(1)
    expect(sources.querySelector('.source.cited .source-title')?.textContent).toBe('Rate Limiting Notes')
    await server.pipe().end()
  })

  it('handles events that arrive split across network chunks', async () => {
    const server = nextStream()
    render(<App />)
    await ask('q')
    const text = sse('token', { text: 'Split ' }) + sse('token', { text: 'words' })
    await server.pipe().send(text.slice(0, 20))
    expect(screen.queryByText('Split words')).toBeNull()
    await server.pipe().send(text.slice(20))
    expect(screen.getByText('Split words')).toBeTruthy()
    await server.pipe().end()
  })

  it('shows "Waiting in queue" while another answer is being generated', async () => {
    const server = nextStream()
    render(<App />)
    await ask('q')
    await server.pipe().send(sse('status', { state: 'queued' }))
    expect(screen.getByText('Waiting in queue...')).toBeTruthy()
    expect(screen.queryByText('Searching your documents...')).toBeNull()

    await server.pipe().send(sse('sources', SOURCES))
    expect(screen.queryByText('Waiting in queue...')).toBeNull()
    expect(screen.getByText(/Reading your sources/)).toBeTruthy() // retrieval is over; the model is reading
    await server.pipe().end()
  })

  it('does not run an HTML tag that appears in an answer', async () => {
    const server = nextStream()
    render(<App />)
    await ask('q')
    await server.pipe().send(sse('token', { text: 'Hello <img src=x onerror=alert(1)> <script>alert(2)</script>' }))
    expect(document.querySelector('img, script')).toBeNull()
    expect(document.body.textContent).toContain('<script>alert(2)</script>')
    await server.pipe().end()
  })
})

describe('compare across documents', () => {
  const OPTION = 'Compare across documents (slower)'

  it('has an option that is unchecked by default', () => {
    render(<App />)
    const option = screen.getByLabelText(OPTION) as HTMLInputElement
    expect(option.type).toBe('checkbox')
    expect(option.checked).toBe(false)
  })

  it('sends no search mode when the option is unchecked, so the server default is used', async () => {
    nextStream()
    render(<App />)
    await ask('How does a token bucket work?')
    expect(askCalls[0].body).toEqual({ question: 'How does a token bucket work?' })
  })

  it('sends the split-and-rerank mode when the option is checked', async () => {
    nextStream()
    render(<App />)
    fireEvent.click(screen.getByLabelText(OPTION))
    expect((screen.getByLabelText(OPTION) as HTMLInputElement).checked).toBe(true)
    await ask('How does a token bucket work, and what is a fixed window?')
    expect(askCalls[0].body).toEqual({
      question: 'How does a token bucket work, and what is a fixed window?',
      mode: 'rerank-split-llm',
    })
    expect(await screen.findByText(/Splitting your question and searching your documents/)).toBeTruthy()
  })

  it('asks again in the same way when a failed question is retried', async () => {
    const server = nextStream()
    render(<App />)
    fireEvent.click(screen.getByLabelText(OPTION))
    await ask('Two things at once?')
    await server.pipe().send(sse('error', { message: 'Something went wrong.' }))
    await server.pipe().end()
    fireEvent.click(screen.getByRole('button', { name: 'Try again' }))
    await waitFor(() => expect(askCalls).toHaveLength(2))
    expect(askCalls[1].body).toEqual({ question: 'Two things at once?', mode: 'rerank-split-llm' })
  })
})

describe('waiting for the first words', () => {
  it('shows the sources as soon as they arrive and a status line about the wait', async () => {
    const server = nextStream()
    render(<App />)
    await ask('How does a token bucket work?')
    expect(await screen.findByText('Searching your documents...')).toBeTruthy()
    expect(screen.queryByRole('region', { name: 'Sources' })).toBeNull()

    await server.pipe().send(sse('sources', SOURCES))
    expect(within(screen.getByRole('region', { name: 'Sources' })).getAllByRole('listitem')).toHaveLength(2)
    const status = screen.getByText(/Reading your sources/).closest('[role="status"]')
    expect(status?.textContent).toBe('Reading your sources - on this CPU the first words can take 30 to 40 seconds')

    await server.pipe().send(sse('token', { text: 'First words' }))
    expect(screen.queryByText(/Reading your sources/)).toBeNull() // the wait is over
    expect(screen.getByRole('region', { name: 'Sources' })).toBeTruthy() // the sources stay
    await server.pipe().end()
  })
})

describe('notices about the answer', () => {
  async function finishWith(done: DoneEvent) {
    const server = nextStream()
    render(<App />)
    await ask('q')
    await server.pipe().send(sse('sources', SOURCES) + sse('done', done))
    await server.pipe().end()
  }

  it('shows a refusal in the not-found style, with no badge, and keeps the text and sources', async () => {
    await finishWith(
      doneEvent({ refused: true, cited: [], uncited: false }, "Unfortunately, I couldn't find this in your documents."),
    )
    expect(screen.getByText(/don't seem to contain an answer/)).toBeTruthy()
    expect(screen.getByText("Unfortunately, I couldn't find this in your documents.")).toBeTruthy()
    expect(screen.getByRole('region', { name: 'Sources' })).toBeTruthy()
    expect(screen.queryByText(/Unverified/)).toBeNull() // a refusal is not "unverified"
  })

  it('shows an "Unverified" badge when an answer that is not a refusal cites no source', async () => {
    await finishWith(doneEvent({ cited: [], uncited: true }, 'A claim with nothing behind it.'))
    expect(screen.getByText('Unverified: this answer cites no source')).toBeTruthy()
    expect(screen.getByText('A claim with nothing behind it.')).toBeTruthy() // the answer stays visible
    expect(screen.getByRole('region', { name: 'Sources' })).toBeTruthy()
    expect(screen.queryByText(/don't seem to contain an answer/)).toBeNull()
    expect(screen.queryByText(/don't match a source/)).toBeNull()
  })

  it('shows a badge when some citation numbers match no source, and keeps the valid ones marked', async () => {
    await finishWith(doneEvent({ cited: [1], invalid_citations: ['0', '7'] }, 'Claim [1] and [0] and [7].'))
    expect(screen.getByText("Some citation numbers don't match a source")).toBeTruthy()
    expect(screen.queryByText(/Unverified/)).toBeNull() // it does cite a real source
    expect(screen.getByText('cited')).toBeTruthy() // source [1] is still marked as cited
  })

  it('can show both badges when the only citations are invalid', async () => {
    await finishWith(doneEvent({ cited: [], uncited: true, invalid_citations: ['9'] }, 'Only a bad one [9].'))
    expect(screen.getByText('Unverified: this answer cites no source')).toBeTruthy()
    expect(screen.getByText("Some citation numbers don't match a source")).toBeTruthy()
  })

  it('shows no badge and no note for a normal, cited answer', async () => {
    await finishWith(doneEvent())
    expect(document.querySelector('.notice')).toBeNull()
    expect(document.querySelector('.flag')).toBeNull()
  })
})

describe('stopping', () => {
  it('stops the request, keeps what was written and gives the box back', async () => {
    const server = nextStream()
    render(<App />)
    await ask('q')
    await server.pipe().send(sse('token', { text: 'Half an ans' }))

    fireEvent.click(screen.getByRole('button', { name: 'Stop' }))

    await waitFor(() => expect(screen.getByText('Stopped.')).toBeTruthy())
    expect(askCalls[0].signal?.aborted).toBe(true)
    expect(screen.getByText('Half an ans')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Send' })).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Stop' })).toBeNull()
    expect(document.querySelector('[role="alert"]')).toBeNull() // stopping is not an error
  })

  it('can stop while still waiting in the queue', async () => {
    const server = nextStream()
    render(<App />)
    await ask('q')
    await server.pipe().send(sse('status', { state: 'queued' }))
    fireEvent.click(screen.getByRole('button', { name: 'Stop' }))
    await waitFor(() => expect(screen.getByText('Stopped.')).toBeTruthy())
    expect(screen.queryByText('Waiting in queue...')).toBeNull()
  })
})

describe('when something goes wrong', () => {
  it('shows the server\'s friendly message with a Try again button that asks again', async () => {
    const server = nextStream()
    render(<App />)
    await ask('How does a token bucket work?')
    await server.pipe().send(sse('sources', SOURCES) + sse('error', { message: 'Could not reach Ollama. Start Ollama and try again.' }))

    const alert = screen.getByRole('alert')
    expect(alert.textContent).toContain('Could not reach Ollama. Start Ollama and try again.')
    expect(alert.textContent).not.toMatch(/Traceback|undefined/)
    await server.pipe().end()

    fireEvent.click(screen.getByRole('button', { name: 'Try again' }))
    await waitFor(() => expect(askCalls).toHaveLength(2))
    expect(askCalls[1].body).toEqual({ question: 'How does a token bucket work?' })
    expect(screen.getAllByText('How does a token bucket work?')).toHaveLength(1) // same turn, not a duplicate
    expect(screen.queryByRole('alert')).toBeNull()
    expect(await screen.findByText('Searching your documents...')).toBeTruthy()

    await server.pipe().send(sse('token', { text: 'It works now' }) + sse('done', doneEvent({}, 'It works now')))
    expect(screen.getByText('It works now')).toBeTruthy()
    await server.pipe().end()
  })

  it('explains a network failure in plain words', async () => {
    onAsk = () => Promise.reject(new TypeError('Failed to fetch'))
    render(<App />)
    await ask('q')
    expect((await screen.findByRole('alert')).textContent).toContain(NETWORK_MESSAGE)
    expect(screen.getByRole('button', { name: 'Try again' })).toBeTruthy()
  })

  it('explains a rejected request without showing technical detail', async () => {
    onAsk = () => Response.json({ detail: [{ type: 'string_too_long', loc: ['body', 'question'] }] }, { status: 422 })
    render(<App />)
    await ask('q')
    const alert = await screen.findByRole('alert')
    expect(alert.textContent).toContain('could not accept that request')
    expect(alert.textContent).not.toContain('string_too_long')
  })

  it('explains a refused request', async () => {
    onAsk = () => Response.json({ detail: 'This origin is not allowed to make changes.' }, { status: 403 })
    render(<App />)
    await ask('q')
    expect((await screen.findByRole('alert')).textContent).toContain('This origin is not allowed')
  })

  it('reports a stream that closes before the answer is finished', async () => {
    const server = nextStream()
    render(<App />)
    await ask('q')
    await server.pipe().send(sse('token', { text: 'Half' }))
    await server.pipe().end()
    expect((await screen.findByRole('alert')).textContent).toContain('stopped before it was finished')
    expect(screen.getByText('Half')).toBeTruthy()
  })

  it('reports an unreadable reply instead of crashing', async () => {
    const server = nextStream()
    render(<App />)
    await ask('q')
    await server.pipe().send('event: token\ndata: not json at all\n\n')
    expect((await screen.findByRole('alert')).textContent).toContain('could not be understood')
    await server.pipe().end()
  })

  it('keeps earlier answers on screen when a later question fails', async () => {
    const first = nextStream()
    render(<App />)
    await ask('First question')
    await first.pipe().send(sse('token', { text: 'First answer' }) + sse('done', doneEvent({}, 'First answer')))
    await first.pipe().end()

    onAsk = () => Promise.reject(new TypeError('Failed to fetch'))
    type('Second question')
    pressEnter()
    await screen.findByRole('alert')
    expect(screen.getByText('First answer')).toBeTruthy()
    expect(screen.getByText('Second question')).toBeTruthy()
    expect(screen.getAllByRole('button', { name: 'Try again' })).toHaveLength(1)
  })
})
