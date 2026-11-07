import { useState } from 'react'
import type { KeyboardEvent } from 'react'

const MAX_QUESTION_LENGTH = 1000

interface Props {
  busy: boolean
  onSend: (question: string) => void
  onStop: () => void
}

export default function Composer({ busy, onSend, onStop }: Props) {
  const [text, setText] = useState('')
  const canSend = !busy && text.trim().length > 0

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
        value={text}
        rows={1}
        maxLength={MAX_QUESTION_LENGTH}
        placeholder="Ask about your documents..."
        onChange={(event) => setText(event.target.value)}
        onKeyDown={onKeyDown}
      />
      {busy ? (
        <button type="button" className="stop" onClick={onStop}>
          Stop
        </button>
      ) : (
        <button type="submit" disabled={!canSend}>
          Send
        </button>
      )}
    </form>
  )
}
