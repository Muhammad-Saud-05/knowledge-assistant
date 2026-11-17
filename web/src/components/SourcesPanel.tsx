import { useEffect, useRef, useState } from 'react'
import type { Turn } from '../chat'
import type { SourceItem } from '../api/types'
import { locationOf, preview } from '../format'
import { CloseIcon } from './Icons'

interface CardProps {
  source: SourceItem
  cited: boolean
  highlighted: boolean
  register: (number: number, element: HTMLLIElement | null) => void
}

function SourceCard({ source, cited, highlighted, register }: CardProps) {
  const [expanded, setExpanded] = useState(false)
  const location = locationOf(source)
  const short = preview(source.text)
  const canExpand = short !== source.text.replace(/\s+/g, ' ').trim()
  const className = ['source', cited ? 'cited' : '', highlighted ? 'highlight' : ''].filter(Boolean).join(' ')
  return (
    <li className={className} ref={(element) => register(source.number, element)} aria-current={highlighted ? 'true' : undefined}>
      <div className="source-head">
        <span className="source-number">[{source.number}]</span>
        <span className="source-title">{source.title ?? 'Untitled'}</span>
        {location && <span className="source-location">{location}</span>}
        {cited && <span className="badge">cited</span>}
        {source.rerank_rank != null && (
          <span className="source-rank">
            rerank #{source.rerank_rank}
            {source.rerank_score != null && ` · score ${source.rerank_score.toFixed(2)}`}
          </span>
        )}
      </div>
      <p className="source-text">{expanded ? source.text : short}</p>
      {canExpand && (
        <button type="button" className="source-more" aria-expanded={expanded} onClick={() => setExpanded(!expanded)}>
          {expanded ? 'Show less' : 'Show more'}
        </button>
      )}
    </li>
  )
}

interface Props {
  turn: Turn
  /** The source to highlight and scroll into view (the one whose chip was clicked), if any. */
  highlight: number | null
  onClose: () => void
}

/** The numbered sources of one answer, in a right-hand panel (a bottom sheet on phones). */
export default function SourcesPanel({ turn, highlight, onClose }: Props) {
  const cited = turn.done?.cited ?? []
  const cards = useRef(new Map<number, HTMLLIElement>())
  const closeButton = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    closeButton.current?.focus()
  }, [])

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [onClose])

  useEffect(() => {
    if (highlight !== null) cards.current.get(highlight)?.scrollIntoView?.({ block: 'nearest' })
  }, [highlight])

  return (
    <>
      <button type="button" className="sheet-backdrop" aria-hidden="true" tabIndex={-1} onClick={onClose} />
      <section className="sources-panel" role="region" aria-label="Sources">
        <div className="panel-head">
          <h2>Sources ({turn.sources.length})</h2>
          <button ref={closeButton} type="button" className="icon-btn" aria-label="Close sources" onClick={onClose}>
            <CloseIcon />
          </button>
        </div>
        <div className="panel-body">
          <ol role="list">
            {turn.sources.map((source) => (
              <SourceCard
                key={source.number}
                source={source}
                cited={cited.includes(source.number)}
                highlighted={highlight === source.number}
                register={(number, element) => {
                  if (element) cards.current.set(number, element)
                  else cards.current.delete(number)
                }}
              />
            ))}
          </ol>
        </div>
      </section>
    </>
  )
}
