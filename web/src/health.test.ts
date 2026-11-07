import { describe, expect, it } from 'vitest'
import type { Health } from './api/types'
import { locationOf, preview } from './format'
import { summariseHealth } from './health'

function health(overrides: Partial<Health> = {}): Health {
  return {
    status: 'ok',
    database: { ok: true, documents: 4, chunks: 26, detail: null },
    ollama: { ok: true, model: 'llama3.2:3b', model_installed: true, detail: null },
    ...overrides,
  }
}

describe('summariseHealth', () => {
  it('is green when everything works', () => {
    expect(summariseHealth(health(), false)).toEqual({ level: 'ok', label: 'Ready - 4 documents, 26 passages' })
  })

  it('explains each kind of problem in plain words', () => {
    expect(summariseHealth(null, true)).toEqual({ level: 'problem', label: "Can't reach the server" })
    expect(summariseHealth(health({ database: { ok: false, documents: null, chunks: null, detail: 'x' } }), false).label).toBe(
      "The database isn't reachable",
    )
    expect(summariseHealth(health({ ollama: { ok: false, model: 'm', model_installed: null, detail: 'x' } }), false).label).toBe(
      "Ollama isn't running",
    )
    expect(summariseHealth(health({ ollama: { ok: true, model: 'llama3.2:3b', model_installed: false, detail: 'x' } }), false).label).toBe(
      "The model llama3.2:3b isn't installed",
    )
  })

  it('says it is checking before the first answer arrives', () => {
    expect(summariseHealth(null, false)).toEqual({ level: 'unknown', label: 'Checking...' })
  })

  it('reports the database problem first when both are down', () => {
    const both = health({
      database: { ok: false, documents: null, chunks: null, detail: null },
      ollama: { ok: false, model: 'm', model_installed: null, detail: null },
    })
    expect(summariseHealth(both, false).label).toBe("The database isn't reachable")
  })
})

describe('source helpers', () => {
  it('cuts a long passage to 150 characters on one line', () => {
    const text = 'word '.repeat(100)
    const shown = preview(text)
    expect(shown.length).toBeLessThanOrEqual(153)
    expect(shown.endsWith('...')).toBe(true)
    expect(shown).not.toContain('\n')
  })

  it('leaves a short passage alone and collapses its whitespace', () => {
    expect(preview('  one\n two\t three  ')).toBe('one two three')
  })

  it('describes where a passage is', () => {
    const base = { number: 1, chunk_id: 1, title: 'T', path: 'p', text: '', vector_rank: null, keyword_rank: null }
    expect(locationOf({ ...base, page_number: 3, section_heading: 'Intro' })).toBe('p. 3 - Intro')
    expect(locationOf({ ...base, page_number: 3, section_heading: null })).toBe('p. 3')
    expect(locationOf({ ...base, page_number: null, section_heading: 'Intro' })).toBe('Intro')
    expect(locationOf({ ...base, page_number: null, section_heading: null })).toBe('')
  })
})
