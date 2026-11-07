import { useEffect, useReducer, useRef } from 'react'
import { ApiError, askQuestion } from './api/client'
import { chatReducer, isActive } from './chat'
import Composer from './components/Composer'
import StatusBadge from './components/StatusBadge'
import TurnView from './components/TurnView'

const UNEXPECTED_MESSAGE = 'Something went wrong while asking. Please try again.'

export default function App() {
  const [turns, dispatch] = useReducer(chatReducer, [])
  const controllerRef = useRef<AbortController | null>(null)
  const nextId = useRef(1)
  const endRef = useRef<HTMLDivElement>(null)
  const busy = turns.some(isActive)

  // Keep the newest words in view as the answer grows.
  useEffect(() => {
    endRef.current?.scrollIntoView?.({ behavior: 'smooth', block: 'end' })
  }, [turns])

  // Leaving the page cancels any answer still being written.
  useEffect(() => () => controllerRef.current?.abort(), [])

  async function run(id: number, question: string) {
    const controller = new AbortController()
    controllerRef.current = controller
    try {
      await askQuestion(
        { question },
        {
          onStatus: (state) => dispatch({ type: 'status', id, state }),
          onSources: (sources) => dispatch({ type: 'sources', id, sources }),
          onToken: (text) => dispatch({ type: 'token', id, text }),
          onDone: (done) => dispatch({ type: 'done', id, done }),
          onError: (message) => dispatch({ type: 'error', id, message }),
        },
        controller.signal,
      )
      // If the stream closed without a final event (and we did not stop it), say so.
      // This does nothing for a turn that already finished.
      if (!controller.signal.aborted) dispatch({ type: 'ended', id })
    } catch (error) {
      dispatch({ type: 'error', id, message: error instanceof ApiError ? error.message : UNEXPECTED_MESSAGE })
    } finally {
      if (controllerRef.current === controller) controllerRef.current = null
    }
  }

  function send(question: string) {
    const id = nextId.current++
    dispatch({ type: 'ask', id, question })
    void run(id, question)
  }

  function stop() {
    controllerRef.current?.abort()
    for (const turn of turns) if (isActive(turn)) dispatch({ type: 'stopped', id: turn.id })
  }

  function retry(id: number) {
    const turn = turns.find((t) => t.id === id)
    if (!turn || busy) return
    dispatch({ type: 'retry', id })
    void run(id, turn.question)
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>Knowledge Assistant</h1>
        <StatusBadge />
      </header>

      <main className="conversation">
        {turns.length === 0 ? (
          <div className="empty">
            <h2>Ask about your documents</h2>
            <p>
              Type a question below. Answers are written only from your own documents, and every claim points to the
              passage it came from. Everything runs on this computer.
            </p>
          </div>
        ) : (
          turns.map((turn) => <TurnView key={turn.id} turn={turn} canRetry={!busy} onRetry={retry} />)
        )}
        <div ref={endRef} />
      </main>

      <Composer busy={busy} onSend={send} onStop={stop} />
    </div>
  )
}
