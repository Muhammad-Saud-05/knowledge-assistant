import { describe, expect, it, vi } from 'vitest'
import { dispatchSseEvent, UNREADABLE_REPLY } from './events'
import type { AskHandlers } from './types'

function handlers() {
  return {
    onStatus: vi.fn(),
    onSources: vi.fn(),
    onToken: vi.fn(),
    onDone: vi.fn(),
    onError: vi.fn(),
  } satisfies AskHandlers
}

const DONE = {
  text: 'Answer [1].',
  validation: { cited_valid: [1], cited_invalid: [], has_no_citations: false, is_not_found: false },
  timings: { retrieval: 0.1, first_token: 0.5, total: 2, queue_wait: 0 },
}

describe('dispatchSseEvent', () => {
  it('passes each known event to its handler', () => {
    const h = handlers()
    dispatchSseEvent({ event: 'status', data: '{"state":"queued"}' }, h)
    dispatchSseEvent({ event: 'sources', data: '[{"number":1,"text":"x"}]' }, h)
    dispatchSseEvent({ event: 'token', data: '{"text":"Hel"}' }, h)
    dispatchSseEvent({ event: 'done', data: JSON.stringify(DONE) }, h)

    expect(h.onStatus).toHaveBeenCalledWith('queued')
    expect(h.onSources).toHaveBeenCalledWith([{ number: 1, text: 'x' }])
    expect(h.onToken).toHaveBeenCalledWith('Hel')
    expect(h.onDone).toHaveBeenCalledWith(DONE)
    expect(h.onError).not.toHaveBeenCalled()
  })

  it('reports an error event through onError', () => {
    const h = handlers()
    dispatchSseEvent({ event: 'error', data: '{"message":"Start Ollama"}' }, h)
    expect(h.onError).toHaveBeenCalledWith('Start Ollama')
  })

  it('keeps newlines and quotes in tokens intact', () => {
    const h = handlers()
    dispatchSseEvent({ event: 'token', data: JSON.stringify({ text: 'a\n"b"' }) }, h)
    expect(h.onToken).toHaveBeenCalledWith('a\n"b"')
  })

  it('ignores an event name it does not know', () => {
    const h = handlers()
    dispatchSseEvent({ event: 'something-new', data: '{"x":1}' }, h)
    dispatchSseEvent({ event: 'something-new', data: 'not even json' }, h)
    for (const fn of Object.values(h)) expect(fn).not.toHaveBeenCalled()
  })

  it.each([
    ['token', 'not json'],
    ['token', '{"text":5}'],
    ['sources', '{"not":"a list"}'],
    ['status', '{}'],
    ['done', '{"text":"x"}'],
    ['done', 'null'],
  ])('reports unreadable %s data (%s) as an error instead of throwing', (event, data) => {
    const h = handlers()
    expect(() => dispatchSseEvent({ event, data }, h)).not.toThrow()
    expect(h.onError).toHaveBeenCalledWith(UNREADABLE_REPLY)
  })

  it('reports an error event with no message as unreadable', () => {
    const h = handlers()
    dispatchSseEvent({ event: 'error', data: '{}' }, h)
    expect(h.onError).toHaveBeenCalledWith(UNREADABLE_REPLY)
  })

  it('works when a handler is missing', () => {
    expect(() => dispatchSseEvent({ event: 'token', data: '{"text":"x"}' }, {})).not.toThrow()
  })
})
