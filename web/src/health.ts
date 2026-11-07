// Turns the answer of /api/health into one short sentence for the status indicator.

import type { Health } from './api/types'

export interface HealthSummary {
  level: 'ok' | 'problem' | 'unknown'
  label: string
}

export function summariseHealth(health: Health | null, serverUnreachable: boolean): HealthSummary {
  if (serverUnreachable) return { level: 'problem', label: "Can't reach the server" }
  if (health === null) return { level: 'unknown', label: 'Checking...' }
  if (!health.database.ok) return { level: 'problem', label: "The database isn't reachable" }
  if (!health.ollama.ok) return { level: 'problem', label: "Ollama isn't running" }
  if (health.ollama.model_installed === false) {
    return { level: 'problem', label: `The model ${health.ollama.model} isn't installed` }
  }
  const { documents, chunks } = health.database
  return { level: 'ok', label: documents === null ? 'Ready' : `Ready - ${documents} documents, ${chunks} passages` }
}
