import { useCallback, useEffect, useRef } from 'react'
import { ApiError, askQuestion } from '../api/client'
import { isActive } from '../chat'
import type { ChatAction } from '../chat'
import { newConversationId, nextTurnId } from '../history'
import { useConversations } from './useConversations'

const UNEXPECTED_MESSAGE = 'Something went wrong while asking. Please try again.'

/**
 * Everything the chat needs: the conversations, the active one, and the actions that start, stop and retry
 * answers. Each answer streams from the server into its own turn; several conversations can be on screen
 * over time, but each turn is answered on its own (the back end keeps no conversation memory).
 */
export function useChat() {
  const { state, dispatch, storageOk } = useConversations()
  const controllers = useRef(new Map<string, AbortController>())
  const stateRef = useRef(state)
  useEffect(() => {
    stateRef.current = state
  })

  const active = state.conversations.find((c) => c.id === state.activeId) ?? null
  const busy = active !== null && active.turns.some(isActive)

  // Leaving the page cancels any answer still being written.
  useEffect(() => {
    const open = controllers.current
    return () => open.forEach((c) => c.abort())
  }, [])

  const run = useCallback(
    async (conversationId: string, turnId: number, question: string, compare: boolean) => {
      const key = `${conversationId}:${turnId}`
      const controller = new AbortController()
      controllers.current.set(key, controller)
      const send = (action: ChatAction) => dispatch({ type: 'turn', conversationId, action })
      try {
        await askQuestion(
          compare ? { question, mode: 'rerank-split-llm' } : { question },
          {
            onStatus: (status) => send({ type: 'status', id: turnId, state: status }),
            onSources: (sources) => send({ type: 'sources', id: turnId, sources }),
            onToken: (text) => send({ type: 'token', id: turnId, text }),
            onDone: (done) => send({ type: 'done', id: turnId, done }),
            onError: (message) => send({ type: 'error', id: turnId, message }),
          },
          controller.signal,
        )
        // If the stream closed without a final event (and we did not stop it), say so.
        // This does nothing for a turn that already finished.
        if (!controller.signal.aborted) send({ type: 'ended', id: turnId })
      } catch (error) {
        send({ type: 'error', id: turnId, message: error instanceof ApiError ? error.message : UNEXPECTED_MESSAGE })
      } finally {
        if (controllers.current.get(key) === controller) controllers.current.delete(key)
      }
    },
    [dispatch],
  )

  const send = useCallback(
    (question: string, compare: boolean) => {
      const current = stateRef.current
      const conversation = current.conversations.find((c) => c.id === current.activeId) ?? null
      const conversationId = conversation?.id ?? newConversationId()
      const turnId = nextTurnId(conversation)
      dispatch({ type: 'ask', conversationId, turnId, question, compare, now: Date.now() })
      void run(conversationId, turnId, question, compare)
    },
    [dispatch, run],
  )

  const stop = useCallback(() => {
    const current = stateRef.current
    const conversation = current.conversations.find((c) => c.id === current.activeId)
    if (!conversation) return
    for (const turn of conversation.turns) {
      if (!isActive(turn)) continue
      controllers.current.get(`${conversation.id}:${turn.id}`)?.abort()
      dispatch({ type: 'turn', conversationId: conversation.id, action: { type: 'stopped', id: turn.id } })
    }
  }, [dispatch])

  /** Ask a turn's question again (after an error, or to regenerate a finished answer). */
  const retry = useCallback(
    (turnId: number) => {
      const current = stateRef.current
      const conversation = current.conversations.find((c) => c.id === current.activeId)
      const turn = conversation?.turns.find((t) => t.id === turnId)
      if (!conversation || !turn || conversation.turns.some(isActive)) return
      dispatch({ type: 'turn', conversationId: conversation.id, action: { type: 'retry', id: turnId } })
      void run(conversation.id, turnId, turn.question, turn.compare)
    },
    [dispatch, run],
  )

  const abortConversation = useCallback((id: string) => {
    for (const [key, controller] of controllers.current) if (key.startsWith(`${id}:`)) controller.abort()
  }, [])

  const newChat = useCallback(() => dispatch({ type: 'new' }), [dispatch])
  const select = useCallback((id: string) => dispatch({ type: 'select', id }), [dispatch])
  const remove = useCallback(
    (id: string) => {
      abortConversation(id)
      dispatch({ type: 'delete', id })
    },
    [abortConversation, dispatch],
  )
  const clearAll = useCallback(() => {
    controllers.current.forEach((c) => c.abort())
    dispatch({ type: 'clear' })
  }, [dispatch])

  return {
    conversations: state.conversations,
    active,
    activeId: state.activeId,
    busy,
    storageOk,
    send,
    stop,
    retry,
    newChat,
    select,
    remove,
    clearAll,
  }
}
