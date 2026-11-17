import { useLayoutEffect, useRef, useState } from 'react'
import type { KeyboardEvent } from 'react'
import { COMPOSER_HINT } from '../texts'
import { SendIcon, StopIcon } from './Icons'

const MAX_QUESTION_LENGTH = 1000

interface Props {
  busy: boolean
  /** "Compare across documents (slower)": split the question into parts and search each. */
  compare: boolean
  onCompareChange: (value: boolean) => void
  onSend: (question: string) => void
  onStop: () => void
}

export default function Composer({ busy, compare, onCompareChange, onSend, onStop }: Props) {
  const [text, setText] = useState('')
  const area = useRef<HTMLTextAreaElement>(null)
  const canSend = !busy && text.trim().length > 0

  // Grow with the text; the stylesheet caps the height at about 8 lines, after which the box scrolls.
  useLayoutEffect(() => {
    const element = area.current
    if (!element) return
    element.style.height = 'auto'
    if (element.scrollHeight > 0) element.style.height = `${element.scrollHeight}px`
  }, [text])

  function submit() {
    if (!canSend) return
    onSend(text.trim())
    setText('')
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    // Enter sends, Shift+Enter adds a new line. Enter while composing text (for example with
    // an input method for Chinese or Japanese) confirms the characters and must not send.
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault()
      submit()
    }
  }

  return (
    <div className="composer-wrap">
      <form
        className="composer"
        onSubmit={(event) => {
          event.preventDefault()
          submit()
        }}
      >
        <label htmlFor="question" className="visually-hidden">
          Your question
        </label>
        <textarea
          id="question"
          ref={area}
          value={text}
          rows={1}
          maxLength={MAX_QUESTION_LENGTH}
          placeholder="Ask about your documents..."
          onChange={(event) => setText(event.target.value)}
          onKeyDown={onKeyDown}
        />
        <div className="composer-row">
          <label className="switch">
            <input type="checkbox" checked={compare} onChange={(event) => onCompareChange(event.target.checked)} />
            Compare across documents (slower)
          </label>
          {busy ? (
            <button type="button" className="send stop" aria-label="Stop" title="Stop" onClick={onStop}>
              <StopIcon />
            </button>
          ) : (
            <button type="submit" className="send" aria-label="Send" title="Send" disabled={!canSend}>
              <SendIcon />
            </button>
          )}
        </div>
      </form>
      <p className="composer-hint">{COMPOSER_HINT}</p>
    </div>
  )
}
