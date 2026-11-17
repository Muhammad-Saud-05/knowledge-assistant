// Conversations: a list of question-and-answer turns shown in one thread. A pure reducer (no React,
// no browser), so it is easy to test. Each turn is still answered on its own: the back end keeps no
// conversation memory, so a conversation is only a way of grouping and remembering what was asked.

import { chatReducer } from './chat'
import type { ChatAction, Turn } from './chat'

/** The most conversations kept; the oldest are dropped first. */
export const MAX_CONVERSATIONS = 50
export const TITLE_CHARS = 60

export interface Conversation {
  id: string
  title: string // the first question, trimmed
  createdAt: number // ms since 1970
  updatedAt: number
  turns: Turn[]
}

export interface HistoryState {
  conversations: Conversation[] // newest first
  activeId: string | null // null: a fresh, empty "new chat"
}

export type HistoryAction =
  | { type: 'new' }
  | { type: 'select'; id: string }
  | { type: 'delete'; id: string }
  | { type: 'clear' }
  | { type: 'ask'; conversationId: string; turnId: number; question: string; compare: boolean; now: number }
  | { type: 'turn'; conversationId: string; action: ChatAction }

export function titleFrom(question: string): string {
  const oneLine = question.replace(/\s+/g, ' ').trim()
  return oneLine.length > TITLE_CHARS ? `${oneLine.slice(0, TITLE_CHARS).trimEnd()}...` : oneLine
}

/** The id for the next turn of a conversation (turn ids only need to be unique within it). */
export function nextTurnId(conversation: Conversation | null | undefined): number {
  return conversation ? Math.max(0, ...conversation.turns.map((t) => t.id)) + 1 : 1
}

export function newConversationId(): string {
  const cryptoApi = typeof globalThis.crypto !== 'undefined' ? globalThis.crypto : undefined
  if (cryptoApi && typeof cryptoApi.randomUUID === 'function') return cryptoApi.randomUUID()
  return `c${Date.now().toString(36)}${Math.random().toString(36).slice(2, 10)}`
}

export function historyReducer(state: HistoryState, action: HistoryAction): HistoryState {
  switch (action.type) {
    case 'new':
      return { ...state, activeId: null }
    case 'select':
      return state.conversations.some((c) => c.id === action.id) ? { ...state, activeId: action.id } : state
    case 'delete': {
      const conversations = state.conversations.filter((c) => c.id !== action.id)
      return { conversations, activeId: state.activeId === action.id ? null : state.activeId }
    }
    case 'clear':
      return { conversations: [], activeId: null }
    case 'ask': {
      const existing = state.conversations.find((c) => c.id === action.conversationId)
      const base: Conversation = existing ?? {
        id: action.conversationId,
        title: titleFrom(action.question),
        createdAt: action.now,
        updatedAt: action.now,
        turns: [],
      }
      const updated: Conversation = {
        ...base,
        updatedAt: action.now,
        turns: chatReducer(base.turns, { type: 'ask', id: action.turnId, question: action.question, compare: action.compare }),
      }
      const others = state.conversations.filter((c) => c.id !== action.conversationId)
      return { conversations: [updated, ...others].slice(0, MAX_CONVERSATIONS), activeId: updated.id }
    }
    case 'turn': {
      if (!state.conversations.some((c) => c.id === action.conversationId)) return state
      return {
        ...state,
        conversations: state.conversations.map((c) =>
          c.id === action.conversationId ? { ...c, turns: chatReducer(c.turns, action.action) } : c,
        ),
      }
    }
  }
}

export type DayGroup = 'Today' | 'Yesterday' | 'Earlier'

function startOfDay(ms: number): number {
  const d = new Date(ms)
  d.setHours(0, 0, 0, 0)
  return d.getTime()
}

export function groupOf(updatedAt: number, now: number): DayGroup {
  const today = startOfDay(now)
  if (updatedAt >= today) return 'Today'
  const yesterday = new Date(today)
  yesterday.setDate(yesterday.getDate() - 1)
  return updatedAt >= yesterday.getTime() ? 'Yesterday' : 'Earlier'
}

/** The conversations split into Today / Yesterday / Earlier (empty groups left out), newest first. */
export function groupConversations(conversations: Conversation[], now: number): { label: DayGroup; items: Conversation[] }[] {
  const order: DayGroup[] = ['Today', 'Yesterday', 'Earlier']
  const sorted = [...conversations].sort((a, b) => b.updatedAt - a.updatedAt)
  return order
    .map((label) => ({ label, items: sorted.filter((c) => groupOf(c.updatedAt, now) === label) }))
    .filter((g) => g.items.length > 0)
}
