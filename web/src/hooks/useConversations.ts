import { useEffect, useReducer, useState } from 'react'
import { isActive } from '../chat'
import { historyReducer } from '../history'
import { loadConversations, saveConversations } from '../storage'

/** The conversations, kept in this browser's storage. If storage is unavailable or full, everything keeps
 *  working in memory and `storageOk` turns false so the page can say history will not be kept. */
export function useConversations() {
  const [loaded] = useState(loadConversations)
  const [state, dispatch] = useReducer(historyReducer, { conversations: loaded.conversations, activeId: null })
  const [storageOk, setStorageOk] = useState(loaded.ok)

  useEffect(() => {
    // Wait until nothing is being written: saving after every streamed word would be wasteful.
    if (state.conversations.some((c) => c.turns.some(isActive))) return
    if (!saveConversations(state.conversations)) queueMicrotask(() => setStorageOk(false))
  }, [state.conversations])

  return { state, dispatch, storageOk }
}
