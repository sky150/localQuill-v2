import { useEffect, useRef, useState } from 'react'
import type { ChangeEvent, DragEvent } from 'react'
import type { DocumentInfo } from '../lib/api'
import type { UploadEntry } from './Sidebar'
import { Bookmark, FileText, Trash, Upload, X } from './icons'

type Tab = 'reference' | 'outline'

type DocumentsModalProps = {
  open: boolean
  documents: DocumentInfo[]
  outline: DocumentInfo | null
  outlinePath: string | null
  uploads: UploadEntry[]
  acceptExtensions: string[]
  files: string[]
  onClose: () => void
  onUpload: (files: File[]) => void
  onDeleteDocument: (docId: string) => void
  onDismissUpload: (id: string) => void
  onSelectOutline: (path: string) => void
  onClearOutline: () => void
}

export default function DocumentsModal({
  open,
  documents,
  outline,
  outlinePath,
  uploads,
  acceptExtensions,
  files,
  onClose,
  onUpload,
  onDeleteDocument,
  onDismissUpload,
  onSelectOutline,
  onClearOutline,
}: DocumentsModalProps) {
  const [tab, setTab] = useState<Tab>('reference')
  const [dragging, setDragging] = useState(false)
  const fileInput = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (!open) return
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])

  if (!open) return null

  const pickFiles = (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files ?? [])
    if (files.length) onUpload(files)
    event.target.value = ''
  }

  const dropFiles = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setDragging(false)
    const files = Array.from(event.dataTransfer.files)
    if (files.length) onUpload(files)
  }

  const radio = (selected: boolean) => (
    <span
      className={`flex h-3.5 w-3.5 shrink-0 items-center justify-center rounded-full border ${
        selected ? 'border-accent' : 'border-base-700'
      }`}
    >
      {selected && <span className="h-1.5 w-1.5 rounded-full bg-accent" />}
    </span>
  )

  const tabs: { id: Tab; label: string }[] = [
    { id: 'reference', label: 'Reference library' },
    { id: 'outline', label: 'Outline' },
  ]

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 p-4"
      onClick={onClose}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Manage documents"
        className="rise flex max-h-[80vh] w-full max-w-lg flex-col rounded-xl border border-line bg-base-850 shadow-2xl"
        onClick={(event) => event.stopPropagation()}
      >
        {/* header */}
        <div className="flex h-12 shrink-0 items-center justify-between border-b border-line px-4">
          <span className="text-[15px] font-semibold tracking-tight text-text">
            Documents
          </span>
          <button
            aria-label="Close"
            title="Close"
            onClick={onClose}
            className="rounded p-1 text-muted hover:bg-base-800 hover:text-text"
          >
            <X size={15} />
          </button>
        </div>

        {/* tabs */}
        <div className="flex shrink-0 gap-1 border-b border-line px-3 pt-2 pb-2">
          {tabs.map((entry) => (
            <button
              key={entry.id}
              onClick={() => setTab(entry.id)}
              aria-pressed={tab === entry.id}
              className={`rounded-md px-2.5 py-1.5 text-[13px] font-medium transition-colors ${
                tab === entry.id
                  ? 'bg-accent-soft text-accent'
                  : 'text-muted hover:bg-base-800 hover:text-text'
              }`}
            >
              {entry.label}
            </button>
          ))}
        </div>

        {/* body */}
        <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3">
          {tab === 'reference' && (
            <>
              <div
                role="button"
                tabIndex={0}
                onClick={() => fileInput.current?.click()}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') fileInput.current?.click()
                }}
                onDragOver={(event) => {
                  if (!event.dataTransfer.types.includes('Files')) return
                  event.preventDefault()
                  setDragging(true)
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={dropFiles}
                className={`cursor-pointer rounded-lg border border-dashed px-3 py-6 text-center text-[12.5px] leading-snug transition-colors ${
                  dragging
                    ? 'border-accent/60 bg-accent-soft text-text'
                    : 'border-line text-muted/70 hover:border-base-700 hover:text-muted'
                }`}
              >
                <Upload size={18} className="mx-auto mb-1.5 opacity-60" />
                Drop documents here, or click to browse
                <span className="mt-1 block text-[11px] text-muted/50">
                  PDF, DOCX, PPTX, XLSX, MD, TXT, CSV, HTML, EPUB
                </span>
              </div>
              <input
                ref={fileInput}
                type="file"
                multiple
                hidden
                accept={acceptExtensions.join(',')}
                onChange={pickFiles}
              />

              <p className="mt-2 px-2 text-center text-[11px] leading-snug text-muted/60">
                Files are processed locally by this Local Quill instance and are
                never shared with third parties.
              </p>

              {uploads.map((entry) => (
                <div
                  key={entry.id}
                  className="mt-2 rounded-md px-2 py-[6px] text-[14px]"
                >
                  <div className="flex items-center gap-2 text-muted">
                    <FileText size={14} className="shrink-0 opacity-60" />
                    <span className="truncate">{entry.name}</span>
                    {entry.status === 'processing' && (
                      <span className="ml-auto shrink-0 animate-pulse text-[10.5px] text-muted">
                        Processing…
                      </span>
                    )}
                    {entry.status === 'error' && (
                      <button
                        aria-label="Dismiss"
                        onClick={() => onDismissUpload(entry.id)}
                        className="ml-auto shrink-0 rounded p-0.5 text-mark hover:bg-base-800"
                      >
                        <X size={12} />
                      </button>
                    )}
                  </div>
                  {entry.status === 'uploading' && (
                    <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-base-800">
                      <div
                        className="h-full rounded-full bg-accent transition-[width] duration-150"
                        style={{ width: `${entry.progress}%` }}
                      />
                    </div>
                  )}
                  {entry.status === 'error' && entry.error && (
                    <p className="mt-0.5 text-[11px] leading-snug text-mark/80">
                      {entry.error}
                    </p>
                  )}
                </div>
              ))}

              {documents.length > 0 && (
                <div className="mt-3 space-y-px">
                  {documents.map((doc) => (
                    <div
                      key={doc.doc_id}
                      title={`${doc.chunks} chunks`}
                      className="group flex items-center gap-2 rounded-md px-2 py-[6px] text-[14px] text-muted hover:bg-base-800 hover:text-text"
                    >
                      <FileText size={14} className="shrink-0 opacity-60" />
                      <span className="truncate">{doc.filename}</span>
                      <button
                        aria-label={`Remove ${doc.filename}`}
                        title={`Remove ${doc.filename}`}
                        onClick={() => onDeleteDocument(doc.doc_id)}
                        className="ml-auto shrink-0 rounded p-0.5 text-muted opacity-0 transition-opacity group-hover:opacity-100 hover:bg-base-700 hover:text-mark"
                      >
                        <Trash size={13} />
                      </button>
                    </div>
                  ))}
                </div>
              )}

              {documents.length === 0 && uploads.length === 0 && (
                <p className="mt-3 px-2 text-[12px] text-muted/50">
                  No reference documents yet. The assistant can draw on anything
                  you add here when answering questions.
                </p>
              )}
            </>
          )}

          {tab === 'outline' && (
            <>
              <p className="px-2 text-[12px] leading-snug text-muted/70">
                The outline is a single file from your workspace that the
                assistant reads to stay aligned with your story structure.
              </p>

              <div className="mt-3 space-y-px">
                <button
                  onClick={onClearOutline}
                  className="flex w-full items-center gap-2.5 rounded-md px-2 py-[7px] text-left text-[14px] text-muted hover:bg-base-800 hover:text-text"
                >
                  {radio(!outline)}
                  <span className={outline ? '' : 'text-text'}>No outline</span>
                </button>

                {files.map((path) => {
                  const selected = outline !== null && path === outlinePath
                  return (
                    <button
                      key={path}
                      onClick={() => onSelectOutline(path)}
                      title={
                        selected
                          ? `${outline.chunks} chunks — click to re-index`
                          : 'Set as outline'
                      }
                      className="flex w-full items-center gap-2.5 rounded-md px-2 py-[7px] text-left text-[14px] text-muted hover:bg-base-800 hover:text-text"
                    >
                      {radio(selected)}
                      <Bookmark
                        size={14}
                        className={`shrink-0 ${selected ? 'text-accent' : 'opacity-40'}`}
                      />
                      <span className={`truncate ${selected ? 'text-text' : ''}`}>
                        {path}
                      </span>
                      {selected && (
                        <span className="ml-auto shrink-0 text-[10.5px] text-accent">
                          Indexed
                        </span>
                      )}
                    </button>
                  )
                })}

                {files.length === 0 && (
                  <p className="px-2 py-[6px] text-[12px] text-muted/50">
                    No text files in your workspace yet.
                  </p>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
