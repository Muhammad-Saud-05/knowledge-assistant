import { describe, expect, it } from 'vitest'
import { ENDED_EARLY_MESSAGE } from './chat'
import type { Turn } from './chat'
import {
  MAX_CONVERSATIONS,
  TITLE_CHARS,
  groupConversations,
  groupOf,
  historyReducer,
  nextTurnId,
  titleFrom,
} from './history'
import type { Conversation, HistoryState } from './history'

const EMPTY: HistoryState = { conversations: [], activeId: null }
const NOW = new Date(2026, 10, 17, 15, 30).getTime() // 17 Nov 2026, 15:30 local time

function ask(state: HistoryState, conversationId: string, question: string, now = NOW): HistoryState {
  const conversation = state.conversations.find((c) => c.id === conversationId)
  return historyReducer(state, { type: 'ask', conversationId, turnId: nextTurnId(conversation), question, compare: false, now })
}

describe('titles', () => {
  it('uses the first question, on one line, trimmed', () => {
    expect(titleFrom('  What is\n a   token bucket?  ')).toBe('What is a token bucket?')
    const long = titleFrom('x'.repeat(200))
    expect(long).toBe(`${'x'.repeat(TITLE_CHARS)}...`)
  })
})

describe('historyReducer', () => {
  it('creates a conversation on the first question and makes it active', () => {
    const state = ask(EMPTY, 'a', 'First question')
    expect(state.activeId).toBe('a')
    expect(state.conversations).toHaveLength(1)
    expect(state.conversations[0]).toMatchObject({ id: 'a', title: 'First question', createdAt: NOW })
    expect(state.conversations[0].turns).toMatchObject([{ id: 1, question: 'First question', status: 'searching' }])
  })

  it('adds later questions to the same conversation as new turns', () => {
    const state = ask(ask(EMPTY, 'a', 'First'), 'a', 'Second')
    expect(state.conversations).toHaveLength(1)
    expect(state.conversations[0].turns.map((t) => [t.id, t.question])).toEqual([[1, 'First'], [2, 'Second']])
    expect(state.conversations[0].title).toBe('First') // the title stays the first question
  })

  it('"new" leaves the active conversation without deleting it; "select" goes back', () => {
    const asked = ask(EMPTY, 'a', 'First')
    const fresh = historyReducer(asked, { type: 'new' })
    expect(fresh.activeId).toBeNull()
    expect(fresh.conversations).toHaveLength(1)
    expect(historyReducer(fresh, { type: 'select', id: 'a' }).activeId).toBe('a')
    expect(historyReducer(fresh, { type: 'select', id: 'missing' })).toBe(fresh)
  })

  it('a question in a fresh chat starts a second conversation, newest first', () => {
    const state = ask(historyReducer(ask(EMPTY, 'a', 'First', NOW - 1000), { type: 'new' }), 'b', 'Second')
    expect(state.conversations.map((c) => c.id)).toEqual(['b', 'a'])
    expect(state.activeId).toBe('b')
  })

  it('deletes one conversation (and leaves the active one if it is another)', () => {
    const both = ask(ask(EMPTY, 'a', 'First'), 'b', 'Second')
    const afterOther = historyReducer(both, { type: 'delete', id: 'a' })
    expect(afterOther.conversations.map((c) => c.id)).toEqual(['b'])
    expect(afterOther.activeId).toBe('b')
    const afterActive = historyReducer(both, { type: 'delete', id: 'b' })
    expect(afterActive.activeId).toBeNull()
  })

  it('clears everything', () => {
    expect(historyReducer(ask(ask(EMPTY, 'a', 'First'), 'b', 'Second'), { type: 'clear' })).toEqual(EMPTY)
  })

  it('applies the usual turn actions to the right turn of the right conversation', () => {
    let state = ask(ask(EMPTY, 'a', 'First'), 'b', 'Second')
    state = historyReducer(state, { type: 'turn', conversationId: 'a', action: { type: 'token', id: 1, text: 'Hello' } })
    const a = state.conversations.find((c) => c.id === 'a')!
    const b = state.conversations.find((c) => c.id === 'b')!
    expect(a.turns[0]).toMatchObject({ status: 'streaming', text: 'Hello' })
    expect(b.turns[0]).toMatchObject({ status: 'searching', text: '' })
    expect(historyReducer(state, { type: 'turn', conversationId: 'gone', action: { type: 'token', id: 1, text: 'x' } })).toBe(state)
  })

  it('keeps at most 50 conversations and drops the oldest', () => {
    let state = EMPTY
    for (let i = 0; i < MAX_CONVERSATIONS + 5; i++) state = ask(historyReducer(state, { type: 'new' }), `c${i}`, `Question ${i}`, NOW + i)
    expect(state.conversations).toHaveLength(MAX_CONVERSATIONS)
    expect(state.conversations[0].id).toBe(`c${MAX_CONVERSATIONS + 4}`) // the newest
    expect(state.conversations.some((c) => c.id === 'c0')).toBe(false) // the oldest are gone
  })
})

describe('nextTurnId', () => {
  it('is one more than the largest id, or 1 for a new conversation', () => {
    expect(nextTurnId(null)).toBe(1)
    expect(nextTurnId({ turns: [{ id: 1 }, { id: 4 }] } as unknown as Conversation)).toBe(5)
  })
})

describe('grouping by day', () => {
  const day = (daysAgo: number, hour = 12) => new Date(2026, 10, 17 - daysAgo, hour).getTime()

  it('puts conversations into Today, Yesterday and Earlier', () => {
    expect(groupOf(day(0, 0), NOW)).toBe('Today')
    expect(groupOf(day(0, 9), NOW)).toBe('Today')
    expect(groupOf(day(1, 23), NOW)).toBe('Yesterday')
    expect(groupOf(day(1, 0), NOW)).toBe('Yesterday')
    expect(groupOf(day(2, 23), NOW)).toBe('Earlier')
    expect(groupOf(day(40), NOW)).toBe('Earlier')
  })

  it('lists the non-empty groups in order, newest first inside each', () => {
    const make = (id: string, at: number): Conversation => ({ id, title: id, createdAt: at, updatedAt: at, turns: [] as Turn[] })
    const groups = groupConversations([make('old', day(10)), make('today-early', day(0, 8)), make('yesterday', day(1)), make('today-late', day(0, 14))], NOW)
    expect(groups.map((g) => g.label)).toEqual(['Today', 'Yesterday', 'Earlier'])
    expect(groups[0].items.map((c) => c.id)).toEqual(['today-late', 'today-early'])
    expect(groupConversations([make('only', day(0))], NOW).map((g) => g.label)).toEqual(['Today'])
    expect(groupConversations([], NOW)).toEqual([])
  })
})

describe('an interrupted turn', () => {
  it('is described by the same message the chat uses for a stream that ended early', () => {
    expect(ENDED_EARLY_MESSAGE).toMatch(/stopped before it was finished/)
  })
})
