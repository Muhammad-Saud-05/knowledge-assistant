import { useEffect, useState } from 'react'
import { getHealth } from '../api/client'
import type { Health } from '../api/types'
import { summariseHealth } from '../health'

const REFRESH_MS = 15_000

/** A small indicator: green when the database and Ollama work, otherwise a short explanation. */
export default function StatusBadge() {
  const [health, setHealth] = useState<Health | null>(null)
  const [unreachable, setUnreachable] = useState(false)

  useEffect(() => {
    const controller = new AbortController()

    async function check() {
      try {
        const result = await getHealth(controller.signal)
        setHealth(result)
        setUnreachable(false)
      } catch {
        if (!controller.signal.aborted) setUnreachable(true)
      }
    }

    void check()
    const timer = window.setInterval(() => void check(), REFRESH_MS)
    return () => {
      controller.abort()
      window.clearInterval(timer)
    }
  }, [])

  const summary = summariseHealth(health, unreachable)
  return (
    <span className={`status status-${summary.level}`} role="status" title={summary.label}>
      <span className="dot" aria-hidden="true" />
      <span className="status-label">{summary.label}</span>
    </span>
  )
}
