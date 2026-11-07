// The state of the conversation, as a pure reducer (no React, no browser).
//
// Each question and its answer is a "turn". The stream from the server is turned into
// actions, and the reducer decides what the turn looks like after each one.

import type { DoneEvent, SourceItem } from './api/types'

export type TurnStatus =
  | 'searching' // sent; waiting for the first piece of the answer
  | 'queued' // another answer is being generated first
  | 'streaming' // the answer is arriving
  | 'done'
  | 'error'
  | 'stopped'

export interface Turn {
  id: number
  question: string
  status: TurnStatus
  text: string
  sources: SourceItem[]
  done: DoneEvent | null
  error: string | null
}

export type ChatAction =
  | { type: 'ask'; id: number; question: string }
  | { type: 'retry'; id: number }
  | { type: 'status'; id: number; state: string }
  | { type: 'sources'; id: number; sources: SourceItem[] }
  | { type: 'token'; id: number; text: string }
  | { type: 'done'; id: number; done: DoneEvent }
  | { type: 'error'; id: number; message: string }
  | { type: 'stopped'; id: number }
  | { type: 'ended'; id: number } // the connection closed without a final event

export const ENDED_EARLY_MESSAGE = 'The answer stopped before it was finished. Please try again.'

const ACTIVE: TurnStatus[] = ['searching', 'queued', 'streaming']

export function isActive(turn: Turn): boolean {
  return ACTIVE.includes(turn.status)
}

function newTurn(id: number, question: string): Turn {
  return { id, question, status: 'searching', text: '', sources: [], done: null, error: null }
}

function update(turns: Turn[], id: number, change: (turn: Turn) => Turn): Turn[] {
  return turns.map((turn) => (turn.id === id ? change(turn) : turn))
}

export function chatReducer(turns: Turn[], action: ChatAction): Turn[] {
  switch (action.type) {
    case 'ask':
      return [...turns, newTurn(action.id, action.question)]
    case 'retry':
      return update(turns, action.id, (turn) => newTurn(turn.id, turn.question))
    default:
      break
  }

  // Everything else describes the progress of an answer. Once a turn is finished,
  // stopped or failed, late events for it (for example a token that was already on its
  // way when Stop was pressed) are ignored.
  return update(turns, action.id, (turn) => {
    if (!isActive(turn)) return turn
    switch (action.type) {
      case 'status':
        return action.state === 'queued' ? { ...turn, status: 'queued' } : turn
      case 'sources':
        // Retrieval has started, so any waiting in the queue is over.
        return { ...turn, sources: action.sources, status: turn.status === 'queued' ? 'searching' : turn.status }
      case 'token':
        return { ...turn, status: 'streaming', text: turn.text + action.text }
      case 'done':
        return { ...turn, status: 'done', text: action.done.text, done: action.done }
      case 'error':
        return { ...turn, status: 'error', error: action.message }
      case 'stopped':
        return { ...turn, status: 'stopped' }
      case 'ended':
        return { ...turn, status: 'error', error: ENDED_EARLY_MESSAGE }
      default:
        return turn
    }
  })
}
