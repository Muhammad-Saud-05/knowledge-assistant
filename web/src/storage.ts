// Saving history and the theme in this browser's localStorage. Every read and write is wrapped in
// try/catch: storage can be missing, blocked (private windows), full or corrupted, and the app must
// keep working in memory when it is.

import { ENDED_EARLY_MESSAGE, isActive } from './chat'
import type { Turn } from './chat'
import { MAX_CONVERSATIONS } from './history'
import type { Conversation } from './history'

export const HISTORY_KEY = 'knowledge-assistant:history:v1'
export const THEME_KEY = 'knowledge-assistant:theme'
/** Source passages are long; keep the stored copy of each to this many characters. */
const STORED_SOURCE_CHARS = 4000

function storage(): Storage | null {
  try {
    return typeof window !== 'undefined' ? window.localStorage : null
  } catch {
    return null
  }
}

export function readSetting(key: string): string | null {
  try {
    return storage()?.getItem(key) ?? null
  } catch {
    return null
  }
}

export function writeSetting(key: string, value: string): boolean {
  try {
    const store = storage()
    if (!store) return false
    store.setItem(key, value)
    return true
  } catch {
    return false
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

/** A turn read back from storage, or null if it is not usable. A turn that was still being answered when the
 *  page closed cannot continue, so it is shown as interrupted. */
function restoreTurn(value: unknown): Turn | null {
  if (!isRecord(value) || typeof value.id !== 'number' || typeof value.question !== 'string') return null
  const turn: Turn = {
    id: value.id,
    question: value.question,
    compare: value.compare === true,
    status: typeof value.status === 'string' ? (value.status as Turn['status']) : 'error',
    text: typeof value.text === 'string' ? value.text : '',
    sources: Array.isArray(value.sources) ? (value.sources as Turn['sources']) : [],
    done: isRecord(value.done) ? (value.done as unknown as Turn['done']) : null,
    error: typeof value.error === 'string' ? value.error : null,
  }
  return isActive(turn) ? { ...turn, status: 'error', error: ENDED_EARLY_MESSAGE } : turn
}

function restoreConversation(value: unknown): Conversation | null {
  if (!isRecord(value) || typeof value.id !== 'string' || typeof value.title !== 'string' || !Array.isArray(value.turns)) {
    return null
  }
  const turns = value.turns.map(restoreTurn).filter((t): t is Turn => t !== null)
  if (turns.length === 0) return null
  return {
    id: value.id,
    title: value.title,
    createdAt: typeof value.createdAt === 'number' ? value.createdAt : 0,
    updatedAt: typeof value.updatedAt === 'number' ? value.updatedAt : 0,
    turns,
  }
}

/** The saved conversations (newest first) and whether storage could be read at all. */
export function loadConversations(): { conversations: Conversation[]; ok: boolean } {
  let raw: string | null
  try {
    const store = storage()
    if (!store) return { conversations: [], ok: false }
    raw = store.getItem(HISTORY_KEY)
  } catch {
    return { conversations: [], ok: false }
  }
  if (!raw) return { conversations: [], ok: true }
  try {
    const data: unknown = JSON.parse(raw)
    const list = isRecord(data) && Array.isArray(data.conversations) ? data.conversations : []
    const conversations = list.map(restoreConversation).filter((c): c is Conversation => c !== null)
    conversations.sort((a, b) => b.updatedAt - a.updatedAt)
    return { conversations: conversations.slice(0, MAX_CONVERSATIONS), ok: true }
  } catch {
    return { conversations: [], ok: true } // unreadable data is ignored, not fatal
  }
}

function forStorage(conversation: Conversation): Conversation {
  return {
    ...conversation,
    turns: conversation.turns.map((turn) => ({
      ...turn,
      sources: turn.sources.map((s) => (s.text.length > STORED_SOURCE_CHARS ? { ...s, text: s.text.slice(0, STORED_SOURCE_CHARS) } : s)),
    })),
  }
}

/** Save the conversations (at most MAX_CONVERSATIONS, newest first). If the browser says the storage is full, the
 *  oldest conversations are dropped until it fits. Returns false only if nothing could be written at all. */
export function saveConversations(conversations: Conversation[]): boolean {
  const store = storage()
  if (!store) return false
  let list = conversations.slice(0, MAX_CONVERSATIONS).map(forStorage)
  for (;;) {
    try {
      store.setItem(HISTORY_KEY, JSON.stringify({ version: 1, conversations: list }))
      return true
    } catch {
      if (list.length === 0) return false
      list = list.slice(0, Math.floor(list.length * 0.8)) // drop the oldest fifth and try again
    }
  }
}
