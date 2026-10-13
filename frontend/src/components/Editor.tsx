import { X } from './icons'

export type OpenFile = { id: string; name: string }

type EditorProps = {
  openFiles: OpenFile[]
  activeId: string | null
  content: string
  wordCount: number
  onSelect: (id: string) => void
  onClose: (id: string) => void
  onChange: (value: string) => void
}

export default function Editor({
  openFiles,
  activeId,
  content,
  wordCount,
  onSelect,
  onClose,
  onChange,
}: EditorProps) {
  const activeName =
    openFiles.find((file) => file.id === activeId)?.name ?? null

  return (
    <section className="rise rise-2 flex h-full min-w-0 flex-1 flex-col border-x border-line bg-base-950">
      {/* open files */}
      {openFiles.length > 0 && (
        <div className="flex h-11 shrink-0 items-end gap-1 overflow-x-auto border-b border-line bg-base-900 px-2">
          {openFiles.map((file) => {
            const isActive = file.id === activeId
            return (
              <div
                key={file.id}
                className={`group flex h-9 items-center gap-1.5 rounded-t-md px-3 text-[13px] ${
                  isActive
                    ? 'border border-b-0 border-line bg-base-950 text-text'
                    : 'text-muted hover:text-text'
                }`}
              >
                <button onClick={() => onSelect(file.id)} className="flex items-center">
                  {file.name}
                </button>
                <button
                  onClick={() => onClose(file.id)}
                  aria-label={`Close ${file.name}`}
                  className={`rounded p-0.5 transition-opacity hover:bg-base-700 ${
                    isActive ? 'opacity-60' : 'opacity-0 group-hover:opacity-60'
                  }`}
                >
                  <X size={12} />
                </button>
              </div>
            )
          })}
        </div>
      )}

      {/* writing surface */}
      {activeId === null ? (
        <div className="flex flex-1 items-center justify-center text-[14px] text-muted">
          Select a file, or create a new one.
        </div>
      ) : (
        <div className="min-h-0 flex-1 overflow-y-auto">
          <div className="mx-auto h-full max-w-[78ch] px-8 py-6">
            <textarea
              value={content}
              onChange={(event) => onChange(event.target.value)}
              placeholder="Start writing…"
              aria-label="File contents"
              className="h-full w-full resize-none bg-transparent text-[15.5px] leading-[1.75] text-text outline-none placeholder:text-muted/60"
            />
          </div>
        </div>
      )}

      {/* status bar */}
      <div className="flex h-9 shrink-0 items-center justify-between gap-3 border-t border-line bg-base-900 px-4">
        <span className="truncate text-[20px] font-semibold text-accent">
          {activeName ?? ''}
        </span>
        {activeId !== null && (
          <span className="shrink-0 text-[20px] font-semibold text-accent">
            {wordCount} {wordCount === 1 ? 'word' : 'words'}
          </span>
        )}
      </div>
    </section>
  )
}
