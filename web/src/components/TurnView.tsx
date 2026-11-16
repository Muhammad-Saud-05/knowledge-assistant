import type { Turn } from '../chat'
import AnswerText from './AnswerText'
import SourceList from './SourceList'

interface Props {
  turn: Turn
  canRetry: boolean
  onRetry: (id: number) => void
}

export const REFUSED_MESSAGE =
  "Your documents don't seem to contain an answer to this. Try rephrasing the question, or add the document that has it."
export const UNVERIFIED_BADGE = 'Unverified: this answer cites no source'
export const INVALID_CITATIONS_BADGE = "Some citation numbers don't match a source"

/** What to say about a finished answer. A refusal is shown as before (a note); an answer that is not a
 *  refusal gets a badge when it cites nothing, and another when some citation matches no source. */
function notes(turn: Turn): { refusal: boolean; badges: string[] } {
  const done = turn.done
  if (!done) return { refusal: false, badges: [] }
  const badges: string[] = []
  if (!done.refused && done.uncited) badges.push(UNVERIFIED_BADGE)
  if (done.invalid_citations.length > 0) badges.push(INVALID_CITATIONS_BADGE)
  return { refusal: done.refused, badges }
}

export const SEARCHING_MESSAGE = 'Searching your documents...'
export const COMPARING_MESSAGE = 'Splitting your question and searching your documents (this takes longer)...'
export const READING_MESSAGE =
  'Reading your sources - on this CPU the first words can take 30 to 40 seconds'

export default function TurnView({ turn, canRetry, onRetry }: Props) {
  const waiting = turn.status === 'searching' || turn.status === 'queued'
  return (
    <article className="turn">
      <div className="question">{turn.question}</div>

      <div className="answer">
        {waiting && (
          <p className="progress" role="status">
            <span className="spinner" aria-hidden="true" />
            {turn.status === 'queued'
              ? 'Waiting in queue...'
              : turn.sources.length > 0
                ? READING_MESSAGE
                : turn.compare
                  ? COMPARING_MESSAGE
                  : SEARCHING_MESSAGE}
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

        {notes(turn).refusal && (
          <p className="notice info" role="note">
            {REFUSED_MESSAGE}
          </p>
        )}

        <SourceList sources={turn.sources} cited={turn.done?.cited ?? []} />

        {notes(turn).badges.length > 0 && (
          <p className="answer-flags">
            {notes(turn).badges.map((text) => (
              <span key={text} className="flag" role="note">
                {text}
              </span>
            ))}
          </p>
        )}

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
