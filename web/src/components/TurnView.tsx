import type { Turn } from '../chat'
import AnswerText from './AnswerText'
import SourceList from './SourceList'

interface Props {
  turn: Turn
  canRetry: boolean
  onRetry: (id: number) => void
}

/** Plain-language warnings about a finished answer. */
function notices(turn: Turn): { kind: 'info' | 'warning'; text: string }[] {
  const done = turn.done
  if (!done) return []
  const found: { kind: 'info' | 'warning'; text: string }[] = []
  if (done.validation.is_not_found) {
    found.push({
      kind: 'info',
      text: "Your documents don't seem to contain an answer to this. Try rephrasing the question, or add the document that has it.",
    })
  }
  if (done.validation.cited_invalid.length > 0) {
    const numbers = done.validation.cited_invalid.map((n) => `[${n}]`).join(', ')
    found.push({
      kind: 'warning',
      text: `The answer cites ${numbers}, which ${done.validation.cited_invalid.length === 1 ? 'is' : 'are'} not among the sources listed. Treat those parts with caution.`,
    })
  }
  if (done.validation.has_no_citations) {
    found.push({
      kind: 'warning',
      text: "This answer has no citations, so it can't be checked against your documents. Treat it with caution.",
    })
  }
  return found
}

export default function TurnView({ turn, canRetry, onRetry }: Props) {
  const waiting = turn.status === 'searching' || turn.status === 'queued'
  return (
    <article className="turn">
      <div className="question">{turn.question}</div>

      <div className="answer">
        {waiting && (
          <p className="progress" role="status">
            <span className="spinner" aria-hidden="true" />
            {turn.status === 'queued' ? 'Waiting in queue...' : 'Searching your documents...'}
          </p>
        )}

        {turn.text && <AnswerText text={turn.text} />}
        {turn.status === 'streaming' && <span className="cursor" aria-hidden="true" />}

        {turn.status === 'stopped' && <p className="stopped">Stopped.</p>}

        {turn.status === 'error' && (
          <div className="error" role="alert">
            <p>{turn.error}</p>
            {canRetry && (
              <button type="button" className="secondary" onClick={() => onRetry(turn.id)}>
                Try again
              </button>
            )}
          </div>
        )}

        {notices(turn).map((notice) => (
          <p key={notice.text} className={`notice ${notice.kind}`} role="note">
            {notice.text}
          </p>
        ))}

        <SourceList sources={turn.sources} cited={turn.done?.validation.cited_valid ?? []} />

        {turn.done && (
          <p className="timing">
            {turn.done.timings.first_token !== null && `First words after ${turn.done.timings.first_token.toFixed(1)}s, `}
            finished in {turn.done.timings.total.toFixed(1)}s
          </p>
        )}
      </div>
    </article>
  )
}
