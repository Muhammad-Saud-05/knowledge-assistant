import { useCallback, useRef, useState } from 'react'
import ChatView from './components/ChatView'
import DocumentsView from './components/DocumentsView'
import { MenuIcon } from './components/Icons'
import Sidebar from './components/Sidebar'
import { useChat } from './hooks/useChat'
import { useMediaQuery } from './hooks/useMediaQuery'
import { useTheme } from './hooks/useTheme'

export default function App() {
  const chat = useChat()
  const { theme, toggleTheme } = useTheme()
  const isMobile = useMediaQuery('(max-width: 767px)')
  const [view, setView] = useState<'chat' | 'documents'>('chat')
  const [compare, setCompare] = useState(false)
  // Changes when the user opens another conversation or a new chat, so the chat view starts afresh. It does NOT change
  // when the first question of a new chat creates its conversation, which would drop the keyboard focus.
  const [chatKey, setChatKey] = useState(0)
  const navigated = useCallback(() => setChatKey((n) => n + 1), [])
  // The sidebar is open on wide screens and closed (a drawer) on phones until the user chooses otherwise.
  const [choice, setChoice] = useState<{ mobile: boolean; open: boolean } | null>(null)
  const open = choice && choice.mobile === isMobile ? choice.open : !isMobile
  const menuButton = useRef<HTMLButtonElement>(null)

  const setOpen = useCallback((value: boolean) => setChoice({ mobile: isMobile, open: value }), [isMobile])
  const closeDrawer = useCallback(() => {
    setOpen(false)
    menuButton.current?.focus() // focus goes back to the button that opened it
  }, [setOpen])
  const afterPick = useCallback(() => {
    if (isMobile) closeDrawer()
  }, [isMobile, closeDrawer])

  const title = view === 'documents' ? 'Documents' : (chat.active?.title ?? 'New chat')

  return (
    <div className="shell">
      <Sidebar
        open={open}
        isMobile={isMobile}
        onClose={closeDrawer}
        conversations={chat.conversations}
        activeId={chat.activeId}
        view={view}
        storageOk={chat.storageOk}
        theme={theme}
        onToggleTheme={toggleTheme}
        onNewChat={() => {
          if (chat.activeId !== null) navigated() // already on a fresh chat: keep what is typed
          chat.newChat()
          setView('chat')
          afterPick()
        }}
        onSelect={(id) => {
          if (id !== chat.activeId) navigated()
          chat.select(id)
          setView('chat')
          afterPick()
        }}
        onDelete={(id) => {
          chat.remove(id)
          navigated()
        }}
        onClearAll={() => {
          chat.clearAll()
          navigated()
        }}
        onShowDocuments={() => {
          setView('documents')
          afterPick()
        }}
      />

      <main className="main">
        <div className="topbar">
          <button
            ref={menuButton}
            type="button"
            className="icon-btn"
            aria-label={open ? 'Hide sidebar' : 'Show sidebar'}
            aria-expanded={open}
            aria-controls="sidebar"
            onClick={() => setOpen(!open)}
          >
            <MenuIcon />
          </button>
          <span className="topbar-title">{title}</span>
        </div>

        {view === 'documents' && (
          <div className="documents-view">
            <DocumentsView />
          </div>
        )}

        {/* The chat stays mounted (just hidden) while the Documents view is open, so a half-typed question is kept. */}
        <div className="view" hidden={view !== 'chat'}>
          <ChatView
            key={chatKey}
            turns={chat.active?.turns ?? []}
            busy={chat.busy}
            compare={compare}
            onCompareChange={setCompare}
            onSend={chat.send}
            onStop={chat.stop}
            onRetry={chat.retry}
          />
        </div>
      </main>
    </div>
  )
}
