import { useLayoutEffect, useRef, useState } from 'react'
import type { Turn } from '../chat'
import { SUGGESTIONS } from '../suggestions'
import { EMPTY_HEADING } from '../texts'
import Composer from './Composer'
import { ArrowDownIcon } from './Icons'
import Message from './Message'
import SourcesPanel from './SourcesPanel'

const NEAR_BOTTOM_PX = 80

interface Props {
  turns: Turn[]
  busy: boolean
  compare: boolean
  onCompareChange: (value: boolean) => void
  onSend: (question: string, compare: boolean) => void
  onStop: () => void
  onRetry: (turnId: number) => void
}

/** One conversation: the messages, the composer, and the sources panel for the answer being looked at. */
export default function ChatView({ turns, busy, compare, onCompareChange, onSend, onStop, onRetry }: Props) {
  const scroller = useRef<HTMLDivElement>(null)
  const atBottom = useRef(true)
  const [away, setAway] = useState(false) // scrolled up, so "Jump to latest" is offered
  const [panel, setPanel] = useState<{ turnId: number; highlight: number | null } | null>(null)
  const trigger = useRef<HTMLElement | null>(null)

  // Follow the newest words, but only while the reader is at the bottom.
  useLayoutEffect(() => {
    const element = scroller.current
    if (element && atBottom.current) element.scrollTop = element.scrollHeight
  }, [turns])

  function onScroll() {
    const element = scroller.current
    if (!element) return
    atBottom.current = element.scrollHeight - element.scrollTop - element.clientHeight < NEAR_BOTTOM_PX
    setAway(!atBottom.current)
  }

  function jumpToLatest() {
    const element = scroller.current
    if (!element) return
    element.scrollTop = element.scrollHeight
    atBottom.current = true
    setAway(false)
  }

  function send(question: string) {
    atBottom.current = true // a new question always scrolls to its answer
    onSend(question, compare)
  }

  function openSources(turnId: number, highlight: number | null, from: HTMLElement) {
    trigger.current = from
    setPanel({ turnId, highlight })
  }

  function closeSources() {
    setPanel(null)
    trigger.current?.focus()
  }

  const panelTurn = panel ? turns.find((t) => t.id === panel.turnId) : undefined

  return (
    <div className="content">
      <div className="chat">
        <div className="thread-area">
        <div className="thread-scroll" ref={scroller} onScroll={onScroll}>
          <div className="thread">
            {turns.length === 0 ? (
              <div className="empty-state">
                <h2>{EMPTY_HEADING}</h2>
                <div className="chips">
                  {SUGGESTIONS.map((suggestion) => (
                    <button key={suggestion} type="button" className="chip" disabled={busy} onClick={() => send(suggestion)}>
                      {suggestion}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              turns.map((turn) => (
                <Message
                  key={turn.id}
                  turn={turn}
                  canRetry={!busy}
                  onRetry={onRetry}
                  onCite={(turnId, number, from) => openSources(turnId, number, from)}
                  onToggleSources={(turnId, from) => (panel?.turnId === turnId ? closeSources() : openSources(turnId, null, from))}
                  sourcesOpen={panel?.turnId === turn.id}
                />
              ))
            )}
          </div>
        </div>
        {away && (
          <button type="button" className="btn btn-small jump-latest" onClick={jumpToLatest}>
            <ArrowDownIcon />
            Jump to latest
          </button>
        )}
        </div>
        <Composer busy={busy} compare={compare} onCompareChange={onCompareChange} onSend={send} onStop={onStop} />
      </div>
      {panel && panelTurn && panelTurn.sources.length > 0 && (
        <SourcesPanel turn={panelTurn} highlight={panel.highlight} onClose={closeSources} />
      )}
    </div>
  )
}
