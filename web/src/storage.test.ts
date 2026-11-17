import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ENDED_EARLY_MESSAGE } from './chat'
import type { Turn } from './chat'
import { MAX_CONVERSATIONS } from './history'
import type { Conversation } from './history'
import { HISTORY_KEY, THEME_KEY, loadConversations, readSetting, saveConversations, writeSetting } from './storage'

function turn(id: number, overrides: Partial<Turn> = {}): Turn {
  return { id, question: `Question ${id}`, compare: false, status: 'done', text: 'An answer.', sources: [], done: null, error: null, ...overrides }
}

function conversation(id: string, updatedAt: number, turns: Turn[] = [turn(1)]): Conversation {
  return { id, title: `Title ${id}`, createdAt: updatedAt, updatedAt, turns }
}

beforeEach(() => localStorage.clear())
afterEach(() => vi.restoreAllMocks())

describe('saving and loading history', () => {
  it('round-trips conversations, newest first', () => {
    expect(saveConversations([conversation('old', 1), conversation('new', 2)])).toBe(true)
    const { conversations, ok } = loadConversations()
    expect(ok).toBe(true)
    expect(conversations.map((c) => c.id)).toEqual(['new', 'old'])
    expect(conversations[0].turns[0]).toMatchObject({ question: 'Question 1', text: 'An answer.' })
  })

  it('starts empty when nothing has been saved', () => {
    expect(loadConversations()).toEqual({ conversations: [], ok: true })
  })

  it('shows a turn that was still being answered when the page closed as interrupted', () => {
    saveConversations([conversation('a', 1, [turn(1, { status: 'streaming', text: 'Half' }), turn(2, { status: 'searching', text: '' })])])
    const [loaded] = loadConversations().conversations
    expect(loaded.turns[0]).toMatchObject({ status: 'error', text: 'Half', error: ENDED_EARLY_MESSAGE })
    expect(loaded.turns[1]).toMatchObject({ status: 'error', error: ENDED_EARLY_MESSAGE })
  })

  it('keeps only the 50 most recent conversations', () => {
    const many = Array.from({ length: MAX_CONVERSATIONS + 10 }, (_, i) => conversation(`c${i}`, 1000 - i))
    saveConversations(many)
    const { conversations } = loadConversations()
    expect(conversations).toHaveLength(MAX_CONVERSATIONS)
    expect(conversations[0].id).toBe('c0')
  })

  it('ignores unreadable or malformed data instead of failing', () => {
    localStorage.setItem(HISTORY_KEY, '{not json')
    expect(loadConversations()).toEqual({ conversations: [], ok: true })
    localStorage.setItem(HISTORY_KEY, JSON.stringify({ conversations: [{ nope: 1 }, 'x', null, { id: 'a', title: 'T', turns: [{ id: 'bad' }] }] }))
    expect(loadConversations().conversations).toEqual([])
  })

  it('stores long source passages in a shortened form', () => {
    const source = { number: 1, chunk_id: 1, title: 'T', path: 'p', page_number: null, section_heading: null, text: 'x'.repeat(10_000), vector_rank: 1, keyword_rank: null }
    saveConversations([conversation('a', 1, [turn(1, { sources: [source] })])])
    expect(loadConversations().conversations[0].turns[0].sources[0].text.length).toBeLessThan(10_000)
  })
})

describe('when storage does not work', () => {
  it('reports that it cannot read and returns nothing, without throwing', () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new Error('blocked')
    })
    expect(loadConversations()).toEqual({ conversations: [], ok: false })
    expect(readSetting(THEME_KEY)).toBeNull()
  })

  it('reports false when nothing can be written', () => {
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('denied')
    })
    expect(saveConversations([conversation('a', 1)])).toBe(false)
    expect(writeSetting(THEME_KEY, 'dark')).toBe(false)
  })

  it('drops the oldest conversations when the storage is full, and keeps the rest', () => {
    const real = Storage.prototype.setItem
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(function (this: Storage, key: string, value: string) {
      if (value.length > 2000) throw new DOMException('full', 'QuotaExceededError')
      real.call(this, key, value)
    })
    const big = (id: string, at: number) => conversation(id, at, [turn(1, { text: 'y'.repeat(600) })])
    const list = Array.from({ length: 10 }, (_, i) => big(`c${i}`, 100 - i))
    expect(saveConversations(list)).toBe(true) // it fits once enough old ones are dropped
    const { conversations } = loadConversations()
    expect(conversations.length).toBeGreaterThan(0)
    expect(conversations.length).toBeLessThan(10)
    expect(conversations[0].id).toBe('c0') // the newest are kept
  })
})

describe('settings', () => {
  it('reads and writes a setting', () => {
    expect(writeSetting(THEME_KEY, 'dark')).toBe(true)
    expect(readSetting(THEME_KEY)).toBe('dark')
  })
})
