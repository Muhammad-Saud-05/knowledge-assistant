import { useCallback, useEffect, useRef, useState } from 'react'
import type { ChangeEvent, DragEvent } from 'react'
import { ApiError, deleteDocument, getDocuments, ingestFile } from '../api/client'
import type { DocumentItem } from '../api/types'
import { EMPTY_MESSAGE, dateOf, describeUpload } from '../documents'
import type { Notice } from '../documents'

const UNEXPECTED = 'Something went wrong. Please try again.'
const MORE_THAN_ONE = ' Only the first file was added; add the others one at a time.'

function messageOf(error: unknown): string {
  return error instanceof ApiError ? error.message : UNEXPECTED
}

/** The Documents page: what is stored, adding files (button or drag and drop) and deleting them. */
export default function DocumentsView() {
  const [documents, setDocuments] = useState<DocumentItem[] | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [uploading, setUploading] = useState<string | null>(null) // the file name being uploaded
  const [notice, setNotice] = useState<Notice | null>(null)
  const [confirming, setConfirming] = useState<number | null>(null) // id waiting for confirmation
  const [deleting, setDeleting] = useState<number | null>(null)
  const [dragging, setDragging] = useState(false)
  const fileInput = useRef<HTMLInputElement>(null)
  const alive = useRef(true)

  useEffect(() => {
    alive.current = true
    return () => {
      alive.current = false
    }
  }, [])

  const refresh = useCallback(async () => {
    try {
      const list = await getDocuments()
      if (!alive.current) return
      setDocuments(list)
      setLoadError(null)
    } catch (error) {
      if (alive.current) setLoadError(messageOf(error))
    }
  }, [])

  useEffect(() => {
    // The first load, started when the page opens.
    async function load() {
      try {
        const list = await getDocuments()
        if (!alive.current) return
        setDocuments(list)
      } catch (error) {
        if (alive.current) setLoadError(messageOf(error))
      }
    }
    void load()
  }, [])

  async function upload(files: FileList | File[] | null | undefined) {
    const list = files ? Array.from(files) : []
    const file = list[0]
    if (!file || uploading !== null) return
    setNotice(null)
    setUploading(file.name)
    try {
      const result = describeUpload(await ingestFile(file))
      if (alive.current) setNotice(list.length > 1 ? { ...result, text: result.text + MORE_THAN_ONE } : result)
    } catch (error) {
      if (alive.current) setNotice({ kind: 'error', text: messageOf(error) })
    } finally {
      if (alive.current) setUploading(null)
    }
    await refresh()
  }

  function onFileChosen(event: ChangeEvent<HTMLInputElement>) {
    const input = event.target
    const files = input.files ? Array.from(input.files) : []
    input.value = '' // so that choosing the same file again is noticed
    void upload(files)
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault()
    setDragging(false)
    void upload(event.dataTransfer.files)
  }

  async function confirmDelete(doc: DocumentItem) {
    setDeleting(doc.id)
    setNotice(null)
    try {
      await deleteDocument(doc.id)
      if (alive.current) setNotice({ kind: 'success', text: `Deleted ${doc.title ?? doc.display_name}.` })
    } catch (error) {
      if (alive.current) setNotice({ kind: 'error', text: messageOf(error) })
    } finally {
      if (alive.current) {
        setDeleting(null)
        setConfirming(null)
      }
    }
    await refresh() // also after a failure, so a document that was already gone disappears
  }

  return (
    <section className="documents" aria-label="Documents">
      <div className="documents-toolbar">
        <h2>Documents</h2>
        <input
          ref={fileInput}
          type="file"
          accept=".pdf,.md"
          className="visually-hidden"
          aria-label="Choose a file to upload"
          tabIndex={-1}
          onChange={onFileChosen}
        />
        <button type="button" className="btn btn-primary" disabled={uploading !== null} onClick={() => fileInput.current?.click()}>
          Upload a document
        </button>
      </div>

      <div
        className={dragging ? 'dropzone dragging' : 'dropzone'}
        data-testid="dropzone"
        onDragEnter={(event) => {
          event.preventDefault()
          setDragging(true)
        }}
        onDragOver={(event) => {
          event.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <p>Drag a PDF or Markdown file here to add it.</p>
        <p>Documents stay on this computer.</p>
      </div>

      {uploading && (
        <p className="progress" role="status">
          <span className="spinner" aria-hidden="true" />
          Uploading and indexing {uploading}... this can take a few seconds.
        </p>
      )}

      {notice && (
        <p className={notice.kind === 'error' ? 'notice warning' : 'notice'} role={notice.kind === 'error' ? 'alert' : 'status'}>
          {notice.text}
        </p>
      )}

      {loadError && (
        <div className="error" role="alert">
          <p>{loadError}</p>
          <button type="button" className="btn btn-small" onClick={() => void refresh()}>
            Try again
          </button>
        </div>
      )}

      {documents === null && !loadError && <p className="progress">Loading your documents...</p>}

      {documents !== null && documents.length === 0 && !loadError && <p className="empty-documents">{EMPTY_MESSAGE}</p>}

      {documents !== null && documents.length > 0 && (
        <div className="table-scroll">
          <table className="documents-table">
            <thead>
              <tr>
                <th scope="col">Title</th>
                <th scope="col">Type</th>
                <th scope="col">Date</th>
                <th scope="col">Chunks</th>
                <th scope="col">
                  <span className="visually-hidden">Actions</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {documents.map((doc) => (
                <tr key={doc.id}>
                  <td>
                    <span className="doc-title">{doc.title ?? doc.display_name}</span>
                    {doc.title && doc.title !== doc.display_name && <span className="doc-file">{doc.display_name}</span>}
                  </td>
                  <td data-label="Type">{doc.source_type === 'pdf' ? 'PDF' : 'Markdown'}</td>
                  <td data-label="Date">{dateOf(doc.doc_date)}</td>
                  <td data-label="Chunks">{doc.chunk_count}</td>
                  <td>
                    {confirming === doc.id ? (
                      <div className="confirm" role="group" aria-label={`Confirm deleting ${doc.title ?? doc.display_name}`}>
                        <p>Delete this document and its {doc.chunk_count} chunks? This cannot be undone.</p>
                        <button type="button" className="btn btn-danger btn-small" disabled={deleting === doc.id} onClick={() => void confirmDelete(doc)}>
                          {deleting === doc.id ? 'Deleting...' : 'Yes, delete'}
                        </button>
                        <button type="button" className="btn btn-small" disabled={deleting === doc.id} onClick={() => setConfirming(null)}>
                          Cancel
                        </button>
                      </div>
                    ) : (
                      <button
                        type="button"
                        className="btn btn-small"
                        aria-label={`Delete ${doc.title ?? doc.display_name}`}
                        disabled={deleting !== null}
                        onClick={() => {
                          setNotice(null)
                          setConfirming(doc.id)
                        }}
                      >
                        Delete
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}
