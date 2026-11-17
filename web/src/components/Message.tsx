import { isActive } from '../chat'
import type { Turn } from '../chat'
import {
  COMPARING_MESSAGE,
  INVALID_CITATIONS_BADGE,
  QUEUED_MESSAGE,
  READING_MESSAGE,
  REFUSED_MESSAGE,
  SEARCHING_MESSAGE,
  UNVERIFIED_BADGE,
} from '../texts'
import AnswerText from './AnswerText'
import CopyButton from './CopyButton'
import { BookIcon, RefreshIcon, SparkIcon } from './Icons'

interface Props {
  turn: Turn
  canRetry: boolean
  onRetry: (id: number) => void
  /** A citation chip in the answer was clicked. */
  onCite: (turnId: number, number: number, trigger: HTMLElement) => void
  /** The "Sources (n)" button was clicked. */
  onToggleSources: (turnId: number, trigger: HTMLElement) => void
  sourcesOpen: boolean
}

/** One question and its answer. */
export default function Message({ turn, canRetry, onRetry, onCite, onToggleSources, sourcesOpen }: Props) {
  const done = turn.done
  const waiting = turn.status === 'searching' || turn.status === 'queued'
  const reading = waiting && turn.status !== 'queued' && turn.sources.length > 0
  const finished = !isActive(turn)
  const hasSources = turn.sources.length > 0

  const waitingText =
    turn.status === 'queued' ? QUEUED_MESSAGE : reading ? READING_MESSAGE : turn.compare ? COMPARING_MESSAGE : SEARCHING_MESSAGE

  const badges: string[] = []
  if (done && !done.refused && done.uncited) badges.push(UNVERIFIED_BADGE)
  if (done && done.invalid_citations.length > 0) badges.push(INVALID_CITATIONS_BADGE)

  return (
    <article className="turn">
      <div className="msg-user">
        <div className="bubble">{turn.question}</div>
      </div>

      <div className="msg-assistant">
        <div className="avatar" aria-hidden="true">
          <SparkIcon />
        </div>
        <div className="msg-body">
          {waiting && (
            <p className={reading ? 'progress reading' : 'progress'} role="status" aria-live="polite">
              <span className="spinner" aria-hidden="true" />
              {waitingText}
            </p>
          )}

          {turn.text && (
            <AnswerText
              text={turn.text}
              sourceCount={turn.sources.length}
              onCite={(number, trigger) => onCite(turn.id, number, trigger)}
              streaming={turn.status === 'streaming'}
            />
          )}

          {turn.status === 'stopped' && <p className="stopped">Stopped.</p>}

          {turn.status === 'error' && (
            <div className="error" role="alert">
              <p>{turn.error}</p>
              {canRetry && (
                <button type="button" className="btn btn-small" onClick={() => onRetry(turn.id)}>
                  Try again
                </button>
              )}
            </div>
          )}

          {done?.refused && (
            <p className="notice" role="note">
              {REFUSED_MESSAGE}
            </p>
          )}

          {badges.length > 0 && (
            <p className="answer-flags">
              {badges.map((text) => (
                <span key={text} className="flag" role="note">
                  {text}
                </span>
              ))}
            </p>
          )}

          {(hasSources || (finished && turn.text)) && (
            <div className="actions">
              {finished && turn.text && <CopyButton text={turn.text} label="Copy answer" caption="Copy" />}
              {finished && turn.text && turn.status !== 'error' && (
                <button type="button" className="action" aria-label="Regenerate answer" disabled={!canRetry} onClick={() => onRetry(turn.id)}>
                  <RefreshIcon />
                  <span>Regenerate</span>
                </button>
              )}
              {hasSources && (
                <button
                  type="button"
                  className="action"
                  aria-expanded={sourcesOpen}
                  onClick={(event) => onToggleSources(turn.id, event.currentTarget)}
                >
                  <BookIcon />
                  <span>Sources ({turn.sources.length})</span>
                </button>
              )}
            </div>
          )}

          {done && (
            <p className="timing">
              {done.timings.first_token !== null && `First words after ${done.timings.first_token.toFixed(1)}s, `}
              finished in {done.timings.total.toFixed(1)}s
            </p>
          )}
        </div>
      </div>
    </article>
  )
}
