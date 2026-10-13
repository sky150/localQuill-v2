import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { api, type ApiConfig, type ChatMessage } from '../lib/api'
import { ArrowUp, AtSign, FileText, X } from './icons'

export type ChatTask = 'qa' | 'grammar' | 'feedback'

type DisplayMessage = ChatMessage & { attachments?: string[] }

const MODES: { label: string; task: ChatTask }[] = [
  { label: 'Q&A', task: 'qa' },
  { label: 'Grammar', task: 'grammar' },
  { label: 'Feedback', task: 'feedback' },
]

type ChatProps = {
  config: ApiConfig | null
  files: string[]
  task: ChatTask
  onTaskChange: (task: ChatTask) => void
  attached: string[]
  onAttach: (path: string) => void
  onDetach: (path: string) => void
}

export default function Chat({
  config,
  files,
  task,
  onTaskChange,
  attached,
  onAttach,
  onDetach,
}: ChatProps) {
  const [messages, setMessages] = useState<DisplayMessage[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [pickerOpen, setPickerOpen] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)

  const mode = MODES.find((entry) => entry.task === task) ?? MODES[0]
  const model = config?.tasks[task]?.model ?? 'model'

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight })
  }, [messages, busy])

  const send = async () => {
    const text = input.trim()
    if ((!text && attached.length === 0) || busy) return
    const history = messages
      .slice(-6)
      .map(({ role, content }) => ({ role, content }))
    setMessages((current) => [
      ...current,
      { role: 'user', content: text, attachments: attached },
    ])
    setInput('')
    setBusy(true)
    try {
      const response = await api.chat({
        task,
        style: 'formal',
        text,
        history,
        contextPaths: attached,
      })
      setMessages((current) => [
        ...current,
        { role: 'assistant', content: response.result },
      ])
    } catch (reason) {
      setMessages((current) => [
        ...current,
        { role: 'assistant', content: `Error: ${(reason as Error).message}` },
      ])
    } finally {
      setBusy(false)
    }
  }

  return (
    <aside className="rise rise-3 flex h-full min-h-0 flex-col bg-base-900">
      {/* header */}
      <div className="flex h-11 shrink-0 items-center justify-between border-b border-line px-4">
        <span className="rounded-full border border-line px-2 py-0.5 text-[12px] text-muted">
          {model}
        </span>
      </div>

      {/* conversation */}
      <div ref={scrollRef} className="min-h-0 flex-1 space-y-6 overflow-y-auto p-4">
        {messages.length === 0 && (
          <p className="text-[13px] leading-relaxed text-muted/70">
            Ask a question about your reference documents, or attach a file to check it
            against your outline.
          </p>
        )}

        {messages.map((message, index) =>
          message.role === 'user' ? (
            <div
              key={index}
              className="ml-auto max-w-[88%] rounded-lg bg-base-800 px-3 py-2"
            >
              {message.attachments && message.attachments.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {message.attachments.map((path) => (
                    <span
                      key={path}
                      className="inline-flex items-center gap-1.5 rounded-md bg-base-700/70 px-1.5 py-0.5 text-[12px] text-text"
                    >
                      <FileText size={12} className="text-muted" />
                      {path.split('/').pop()}
                    </span>
                  ))}
                </div>
              )}
              {message.content && (
                <p
                  className={`text-[14px] whitespace-pre-wrap text-text ${
                    message.attachments?.length ? 'mt-1.5' : ''
                  }`}
                >
                  {message.content}
                </p>
              )}
            </div>
          ) : (
            <div key={index}>
              <div className="mb-1.5 text-[12px] font-medium text-muted">Quill</div>
              <div className="chat-markdown">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {message.content}
                </ReactMarkdown>
              </div>
            </div>
          ),
        )}

        {busy && (
          <div>
            <div className="mb-1.5 text-[12px] font-medium text-muted">Quill</div>
            <p className="animate-pulse text-[14px] text-muted">Thinking…</p>
          </div>
        )}
      </div>

      {/* composer */}
      <div className="relative shrink-0 border-t border-line p-3">
        {pickerOpen && (
          <div className="absolute bottom-full left-3 z-10 mb-2 max-h-64 w-72 overflow-y-auto rounded-lg border border-line bg-base-850 p-1 shadow-lg">
            {files.filter((path) => !attached.includes(path)).length === 0 && (
              <p className="px-2 py-1.5 text-[12px] text-muted/60">
                No files to attach
              </p>
            )}
            {files
              .filter((path) => !attached.includes(path))
              .map((path) => (
                <button
                  key={path}
                  onClick={() => {
                    onAttach(path)
                    setPickerOpen(false)
                  }}
                  className="block w-full truncate rounded-md px-2 py-1.5 text-left text-[13px] text-muted hover:bg-base-800 hover:text-text"
                >
                  {path}
                </button>
              ))}
          </div>
        )}

        <div className="rounded-xl border border-line bg-base-850 transition-colors focus-within:border-accent/50">
          {attached.length > 0 && (
            <div className="flex flex-wrap gap-1.5 px-2.5 pt-2.5">
              {attached.map((path) => (
                <span
                  key={path}
                  className="flex max-w-full items-center gap-1.5 rounded-md bg-base-700/70 py-1 pr-1 pl-2 text-[12px] text-text"
                >
                  <FileText size={12} className="shrink-0 text-muted" />
                  <span className="truncate">{path.split('/').pop()}</span>
                  <button
                    aria-label={`Remove ${path}`}
                    onClick={() => onDetach(path)}
                    className="shrink-0 rounded p-0.5 text-muted hover:text-text"
                  >
                    <X size={12} />
                  </button>
                </span>
              ))}
            </div>
          )}

          <textarea
            rows={2}
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault()
                void send()
              }
            }}
            placeholder={
              task === 'qa'
                ? 'Ask about your documents…'
                : attached.length > 0
                  ? 'Optional note — the attached file will be reviewed'
                  : `Paste text for ${mode.label.toLowerCase()}…`
            }
            className="mt-1 w-full resize-none bg-transparent px-3 py-1.5 text-[14px] text-text outline-none placeholder:text-muted/50"
          />

          <div className="flex items-center gap-1.5 px-2 pb-2">
            <button
              aria-label="Add context"
              title="Add context file"
              onClick={() => setPickerOpen((open) => !open)}
              className={`flex size-7 shrink-0 items-center justify-center rounded-full border text-muted transition-colors hover:text-text ${
                pickerOpen
                  ? 'border-accent/60 text-text'
                  : 'border-line hover:border-base-700'
              }`}
            >
              <AtSign size={13} />
            </button>
            <div
              className="flex min-w-0 rounded-full border border-line p-0.5"
              role="group"
              aria-label="Assistant mode"
            >
              {MODES.map((m) => (
                <button
                  key={m.task}
                  onClick={() => onTaskChange(m.task)}
                  aria-pressed={m.task === task}
                  className={`rounded-full px-2.5 py-1 text-[12px] whitespace-nowrap transition-colors ${
                    m.task === task
                      ? 'bg-accent font-semibold text-base-950'
                      : 'text-muted hover:text-text'
                  }`}
                >
                  {m.label}
                </button>
              ))}
            </div>
            <button
              onClick={() => void send()}
              disabled={busy}
              aria-label="Send"
              className="ml-auto flex size-7 shrink-0 items-center justify-center rounded-full bg-accent text-base-950 transition-colors hover:bg-accent-bright disabled:opacity-40"
            >
              <ArrowUp size={14} />
            </button>
          </div>
        </div>
      </div>
    </aside>
  )
}
