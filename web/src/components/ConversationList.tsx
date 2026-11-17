import { groupConversations } from '../history'
import type { Conversation } from '../history'
import { TrashIcon } from './Icons'

interface Props {
  conversations: Conversation[]
  activeId: string | null
  /** The chat is the visible view (so one of the conversations can be "the active one"). */
  chatShown: boolean
  now: number
  onSelect: (id: string) => void
  onDelete: (id: string) => void
}

/** The history list, grouped into Today / Yesterday / Earlier. */
export default function ConversationList({ conversations, activeId, chatShown, now, onSelect, onDelete }: Props) {
  if (conversations.length === 0) return <p className="conv-empty">Your conversations will appear here.</p>
  return (
    <>
      {groupConversations(conversations, now).map((group) => (
        <section key={group.label} className="conv-group" aria-label={group.label}>
          <h2>{group.label}</h2>
          <ul>
            {group.items.map((conversation) => {
              const active = chatShown && conversation.id === activeId
              return (
                <li key={conversation.id} className={active ? 'conv-item active' : 'conv-item'}>
                  <button
                    type="button"
                    className="conv-open"
                    aria-current={active ? 'true' : undefined}
                    title={conversation.title}
                    onClick={() => onSelect(conversation.id)}
                  >
                    {conversation.title}
                  </button>
                  <button
                    type="button"
                    className="icon-btn conv-delete"
                    aria-label={`Delete conversation: ${conversation.title}`}
                    title="Delete"
                    onClick={() => onDelete(conversation.id)}
                  >
                    <TrashIcon />
                  </button>
                </li>
              )
            })}
          </ul>
        </section>
      ))}
    </>
  )
}
