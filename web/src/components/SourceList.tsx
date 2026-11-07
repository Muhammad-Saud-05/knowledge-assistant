import type { SourceItem } from '../api/types'
import { locationOf, preview } from '../format'

interface Props {
  sources: SourceItem[]
  /** Numbers of the sources the answer actually cites (known once it is finished). */
  cited: number[]
}

export default function SourceList({ sources, cited }: Props) {
  if (sources.length === 0) return null
  return (
    <section className="sources" aria-label="Sources">
      <h3>Sources</h3>
      <ol role="list">
        {sources.map((source) => {
          const location = locationOf(source)
          const isCited = cited.includes(source.number)
          return (
            <li key={source.number} className={isCited ? 'source cited' : 'source'}>
              <div className="source-head">
                <span className="source-number">[{source.number}]</span>
                <span className="source-title">{source.title ?? 'Untitled'}</span>
                {location && <span className="source-location">{location}</span>}
                {isCited && <span className="badge">cited</span>}
              </div>
              <p className="source-text">{preview(source.text)}</p>
            </li>
          )
        })}
      </ol>
    </section>
  )
}
