import { describe, expect, it } from 'vitest'
import type { DoneEvent, SourceItem } from './api/types'
import { chatReducer, ENDED_EARLY_MESSAGE, isActive, type ChatAction, type Turn } from './chat'

const SOURCE: SourceItem = {
  number: 1,
  chunk_id: 5,
  title: 'Notes',
  path: 'notes.md',
  page_number: null,
  section_heading: 'Token bucket',
  text: 'A bucket holds tokens.',
  vector_rank: 1,
  keyword_rank: null,
}

const DONE: DoneEvent = {
  text: 'Final text [1].',
  validation: { cited_valid: [1], cited_invalid: [], has_no_citations: false, is_not_found: false },
  timings: { retrieval: 0.1, first_token: 0.4, total: 1.2, queue_wait: 0 },
}

function run(...actions: ChatAction[]): Turn[] {
  return actions.reduce<Turn[]>(chatReducer, [])
}

const ASK: ChatAction = { type: 'ask', id: 1, question: 'What is a token bucket?' }

describe('chatReducer', () => {
  it('starts a turn in the searching state', () => {
    const [turn] = run(ASK)
    expect(turn).toMatchObject({ id: 1, question: 'What is a token bucket?', status: 'searching', text: '', sources: [], done: null, error: null })
    expect(isActive(turn)).toBe(true)
  })

  it('shows the queue state and leaves it when retrieval starts', () => {
    const queued = run(ASK, { type: 'status', id: 1, state: 'queued' })
    expect(queued[0].status).toBe('queued')
    const afterSources = chatReducer(queued, { type: 'sources', id: 1, sources: [SOURCE] })
    expect(afterSources[0].status).toBe('searching')
    expect(afterSources[0].sources).toEqual([SOURCE])
  })

  it('ignores a status it does not know', () => {
    expect(run(ASK, { type: 'status', id: 1, state: 'warming-up' })[0].status).toBe('searching')
  })

  it('adds up tokens and switches to streaming', () => {
    const [turn] = run(ASK, { type: 'sources', id: 1, sources: [SOURCE] }, { type: 'token', id: 1, text: 'A buc' }, { type: 'token', id: 1, text: 'ket' })
    expect(turn.status).toBe('streaming')
    expect(turn.text).toBe('A bucket')
  })

  it('finishes with the authoritative full text from the done event', () => {
    const [turn] = run(ASK, { type: 'token', id: 1, text: 'partial' }, { type: 'done', id: 1, done: DONE })
    expect(turn.status).toBe('done')
    expect(turn.text).toBe('Final text [1].')
    expect(turn.done).toEqual(DONE)
    expect(isActive(turn)).toBe(false)
  })

  it('records an error but keeps what was already written', () => {
    const [turn] = run(ASK, { type: 'token', id: 1, text: 'partial' }, { type: 'error', id: 1, message: 'Start Ollama' })
    expect(turn).toMatchObject({ status: 'error', error: 'Start Ollama', text: 'partial' })
  })

  it('ignores events that arrive after Stop', () => {
    const [turn] = run(ASK, { type: 'token', id: 1, text: 'a' }, { type: 'stopped', id: 1 }, { type: 'token', id: 1, text: 'late' }, { type: 'done', id: 1, done: DONE })
    expect(turn.status).toBe('stopped')
    expect(turn.text).toBe('a')
    expect(turn.done).toBeNull()
  })

  it('does not let a late "ended" turn a finished answer into an error', () => {
    const [turn] = run(ASK, { type: 'done', id: 1, done: DONE }, { type: 'ended', id: 1 })
    expect(turn.status).toBe('done')
  })

  it('turns a stream that closed early into a plain-language error', () => {
    const [turn] = run(ASK, { type: 'token', id: 1, text: 'half' }, { type: 'ended', id: 1 })
    expect(turn).toMatchObject({ status: 'error', error: ENDED_EARLY_MESSAGE, text: 'half' })
  })

  it('resets a turn on retry but keeps its question and position', () => {
    const turns = run(ASK, { type: 'ask', id: 2, question: 'Second' }, { type: 'token', id: 1, text: 'x' }, { type: 'error', id: 1, message: 'boom' }, { type: 'retry', id: 1 })
    expect(turns.map((t) => t.id)).toEqual([1, 2])
    expect(turns[0]).toMatchObject({ question: 'What is a token bucket?', status: 'searching', text: '', error: null, sources: [] })
    expect(turns[1].question).toBe('Second')
  })

  it('only changes the turn an event is for', () => {
    const turns = run(ASK, { type: 'ask', id: 2, question: 'Second' }, { type: 'token', id: 2, text: 'only second' })
    expect(turns[0].text).toBe('')
    expect(turns[1].text).toBe('only second')
  })
})
