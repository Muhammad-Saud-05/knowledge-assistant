// Turns parsed stream events into calls on the handlers. Pure logic, no browser dependencies.

import type { SseEvent } from './sse'
import type { AskHandlers, DoneEvent, SourceItem } from './types'

export const UNREADABLE_REPLY = 'The server sent a reply that could not be understood. Please try again.'

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/**
 * Handle one event. Events with a name we do not know are ignored, so the server can add
 * new kinds later without breaking this app. A known event with unreadable data is
 * reported through onError instead of throwing.
 */
export function dispatchSseEvent(event: SseEvent, handlers: AskHandlers): void {
  let payload: unknown
  try {
    payload = JSON.parse(event.data)
  } catch {
    if (['status', 'sources', 'token', 'done', 'error'].includes(event.event)) {
      handlers.onError?.(UNREADABLE_REPLY)
    }
    return
  }

  switch (event.event) {
    case 'status':
      if (isObject(payload) && typeof payload.state === 'string') handlers.onStatus?.(payload.state)
      else handlers.onError?.(UNREADABLE_REPLY)
      break
    case 'sources':
      if (Array.isArray(payload)) handlers.onSources?.(payload as SourceItem[])
      else handlers.onError?.(UNREADABLE_REPLY)
      break
    case 'token':
      if (isObject(payload) && typeof payload.text === 'string') handlers.onToken?.(payload.text)
      else handlers.onError?.(UNREADABLE_REPLY)
      break
    case 'done':
      if (isObject(payload) && typeof payload.text === 'string' && isObject(payload.validation) && isObject(payload.timings)) {
        handlers.onDone?.(payload as unknown as DoneEvent)
      } else {
        handlers.onError?.(UNREADABLE_REPLY)
      }
      break
    case 'error':
      handlers.onError?.(isObject(payload) && typeof payload.message === 'string' ? payload.message : UNREADABLE_REPLY)
      break
    default:
      break // an event we do not know about
  }
}
