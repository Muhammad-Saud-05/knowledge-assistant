// Text for the Documents page, kept apart from the component so it can be tested on its own.

import type { IngestResult } from './api/types'

export const EMPTY_MESSAGE = 'No documents yet. Upload a PDF or Markdown file to get started.'

export type Notice = { kind: 'success' | 'error'; text: string }

/** What to tell the user about a finished upload. */
export function describeUpload(result: IngestResult): Notice {
  const name = result.filename
  switch (result.status) {
    case 'new':
      return { kind: 'success', text: `Added ${name} (${result.chunks} passages).` }
    case 'updated':
      return { kind: 'success', text: `Updated ${name} (${result.chunks} passages).` }
    case 'unchanged':
      return { kind: 'success', text: `${name} is already stored and has not changed.` }
    default:
      return { kind: 'error', text: result.error ?? `${name} could not be added.` }
  }
}

/** A date as YYYY-MM-DD, or "-" when the document has none. */
export function dateOf(value: string | null): string {
  return value ? value.slice(0, 10) : '-'
}
