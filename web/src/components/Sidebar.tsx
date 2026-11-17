import { useEffect, useMemo, useState } from 'react'
import type { Conversation } from '../history'
import type { Theme } from '../hooks/useTheme'
import ConversationList from './ConversationList'
import { CloseIcon, FileIcon, PlusIcon, TrashIcon } from './Icons'
import StatusBadge from './StatusBadge'
import ThemeToggle from './ThemeToggle'

interface Props {
  open: boolean
  /** Phone layout: the sidebar is a drawer over the page. */
  isMobile: boolean
  onClose: () => void
  conversations: Conversation[]
  activeId: string | null
  view: 'chat' | 'documents'
  storageOk: boolean
  theme: Theme
  onToggleTheme: () => void
  onNewChat: () => void
  onSelect: (id: string) => void
  onDelete: (id: string) => void
  onClearAll: () => void
  onShowDocuments: () => void
}

export default function Sidebar(props: Props) {
  const { open, isMobile, onClose, conversations, activeId, view, storageOk, theme } = props
  const [confirmingClear, setConfirmingClear] = useState(false)
  // "Today" and "Yesterday" are worked out again whenever the list changes.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const now = useMemo(() => Date.now(), [conversations])

  // Esc closes the drawer.
  useEffect(() => {
    if (!isMobile || !open) return
    function onKey(event: KeyboardEvent) {
      if (event.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [isMobile, open, onClose])

  const className = ['sidebar', open ? 'open' : isMobile ? '' : 'collapsed'].filter(Boolean).join(' ')

  return (
    <>
      {isMobile && open && <button type="button" className="backdrop" aria-hidden="true" tabIndex={-1} onClick={onClose} />}
      <aside id="sidebar" className={className} aria-label="Sidebar" inert={!open}>
        <div className="sidebar-head">
          <h1 className="app-name">Knowledge Assistant</h1>
          {isMobile && (
            <button type="button" className="icon-btn" aria-label="Close menu" onClick={onClose}>
              <CloseIcon />
            </button>
          )}
        </div>

        <button type="button" className="btn btn-primary new-chat" onClick={props.onNewChat}>
          <PlusIcon />
          New chat
        </button>

        <nav className="conv-scroll" aria-label="Conversations">
          <ConversationList
            conversations={conversations}
            activeId={activeId}
            chatShown={view === 'chat'}
            now={now}
            onSelect={props.onSelect}
            onDelete={props.onDelete}
          />
        </nav>

        <div className="sidebar-foot">
          <button type="button" className="foot-item" aria-current={view === 'documents' ? 'page' : undefined} onClick={props.onShowDocuments}>
            <FileIcon />
            <span>Documents</span>
          </button>
          <ThemeToggle theme={theme} onToggle={props.onToggleTheme} />
          {confirmingClear ? (
            <div className="foot-confirm" role="group" aria-label="Confirm clearing history">
              <p>Delete all conversations from this browser? This cannot be undone.</p>
              <button
                type="button"
                className="btn btn-danger btn-small"
                onClick={() => {
                  props.onClearAll()
                  setConfirmingClear(false)
                }}
              >
                Yes, clear all
              </button>
              <button type="button" className="btn btn-small" onClick={() => setConfirmingClear(false)}>
                Cancel
              </button>
            </div>
          ) : (
            <button type="button" className="foot-item" disabled={conversations.length === 0} onClick={() => setConfirmingClear(true)}>
              <TrashIcon />
              <span>Clear all history</span>
            </button>
          )}
          <p className="foot-note">
            {storageOk
              ? 'History stays on this device.'
              : "History can't be saved in this browser, so it is kept only until you close this page."}
          </p>
          <div className="sidebar-health">
            <StatusBadge />
          </div>
        </div>
      </aside>
    </>
  )
}
