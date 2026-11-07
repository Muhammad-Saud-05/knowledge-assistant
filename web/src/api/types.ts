// The shapes of the data exchanged with the back end (see src/api/schemas.py and ask.py).

export type SourceType = 'pdf' | 'markdown'
export type SearchMode = 'vector' | 'keyword' | 'hybrid'

export interface Filters {
  source_types?: SourceType[]
  date_from?: string // YYYY-MM-DD
  date_to?: string
  path_prefix?: string
}

export interface AskRequest {
  question: string
  k?: number
  mode?: SearchMode
  filters?: Filters
}

export interface SourceItem {
  number: number
  chunk_id: number
  title: string | null
  path: string
  page_number: number | null
  section_heading: string | null
  text: string
  vector_rank: number | null
  keyword_rank: number | null
}

export interface Validation {
  cited_valid: number[]
  cited_invalid: number[]
  has_no_citations: boolean
  is_not_found: boolean
}

export interface Timings {
  retrieval: number
  first_token: number | null
  total: number
  queue_wait: number
}

export interface DoneEvent {
  text: string
  validation: Validation
  timings: Timings
}

/** What to do for each kind of event in an answer stream. */
export interface AskHandlers {
  onStatus?: (state: string) => void
  onSources?: (sources: SourceItem[]) => void
  onToken?: (text: string) => void
  onDone?: (done: DoneEvent) => void
  onError?: (message: string) => void
}

export interface DatabaseHealth {
  ok: boolean
  documents: number | null
  chunks: number | null
  detail: string | null
}

export interface OllamaHealth {
  ok: boolean
  model: string
  model_installed: boolean | null
  detail: string | null
}

export interface Health {
  status: 'ok' | 'degraded'
  database: DatabaseHealth
  ollama: OllamaHealth
}

export interface DocumentItem {
  id: number
  title: string | null
  path: string
  source_type: SourceType
  doc_date: string | null
  ingested_at: string
  chunks: number
}

export interface IngestResult {
  status: 'new' | 'updated' | 'unchanged' | 'failed'
  chunks: number
  document_id: number | null
  filename: string
  error: string | null
}

export interface DeleteResult {
  deleted: boolean
  file_removed: boolean
}
