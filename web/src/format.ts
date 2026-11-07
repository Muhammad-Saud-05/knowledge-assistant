// Small text helpers for showing sources.

import type { SourceItem } from './api/types'

export const PREVIEW_CHARS = 150

/** The first `limit` characters of a passage on one line, with "..." if it was cut. */
export function preview(text: string, limit: number = PREVIEW_CHARS): string {
  const oneLine = text.replace(/\s+/g, ' ').trim()
  return oneLine.length > limit ? `${oneLine.slice(0, limit).trimEnd()}...` : oneLine
}

/** Where in the document a passage is: "p. 3 - Heading", or just one of them. */
export function locationOf(source: SourceItem): string {
  const parts: string[] = []
  if (source.page_number !== null) parts.push(`p. ${source.page_number}`)
  if (source.section_heading) parts.push(source.section_heading)
  return parts.join(' - ')
}
