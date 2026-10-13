import { useRef, useState } from 'react'
import type { ChangeEvent, DragEvent } from 'react'
import type { DocumentInfo, FileNode } from '../lib/api'
import { Bookmark, ChevronRight, FileText, FolderPlus, Plus, Trash, X } from './icons'

export type UploadEntry = {
  id: string
  name: string
  progress: number
  status: 'uploading' | 'processing' | 'error'
  error?: string
}

type SidebarProps = {
  tree: FileNode[]
  documents: DocumentInfo[]
  outline: DocumentInfo | null
  uploads: UploadEntry[]
  activeId: string | null
  editingId: string | null
  acceptExtensions: string[]
  onOpen: (path: string) => void
  onCreateFile: () => void
  onCreateFolder: () => void
  onDelete: (path: string) => void
  onRename: (path: string, value: string) => void
  onMove: (path: string, parent: string) => void
  onUpload: (files: File[]) => void
  onSetOutline: (path: string) => void
  onClearOutline: () => void
  onDismissUpload: (id: string) => void
}

export default function Sidebar({
  tree,
  documents,
  outline,
  uploads,
  activeId,
  editingId,
  acceptExtensions,
  onOpen,
  onCreateFile,
  onCreateFolder,
  onDelete,
  onRename,
  onMove,
  onUpload,
  onSetOutline,
  onClearOutline,
  onDismissUpload,
}: SidebarProps) {
  const [open, setOpen] = useState<ReadonlySet<string>>(new Set())
  const [dragging, setDragging] = useState(false)
  const [dragPath, setDragPath] = useState<string | null>(null)
  const [dropTarget, setDropTarget] = useState<string | null>(null)
  const fileInput = useRef<HTMLInputElement>(null)

  const toggle = (path: string) =>
    setOpen((prev) => {
      const next = new Set(prev)
      if (next.has(path)) {
        next.delete(path)
      } else {
        next.add(path)
      }
      return next
    })

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

  const actions = (node: FileNode) => (
    <div className="absolute top-1/2 right-1 flex -translate-y-1/2 items-center gap-0.5 opacity-0 transition-opacity group-hover:opacity-100">
      {node.type === 'file' && /\.(md|markdown|txt)$/i.test(node.name) && (
        <button
          aria-label={`Set ${node.name} as outline`}
          title="Set as outline"
          onClick={(event) => {
            event.stopPropagation()
            onSetOutline(node.path)
          }}
          className="rounded bg-base-800 p-1 text-muted hover:bg-base-700 hover:text-text"
        >
          <Bookmark size={13} />
        </button>
      )}
      <button
        aria-label={`Delete ${node.name}`}
        title={`Delete ${node.name}`}
        onClick={(event) => {
          event.stopPropagation()
          onDelete(node.path)
        }}
        className="rounded bg-base-800 p-1 text-muted hover:bg-base-700 hover:text-text"
      >
        <Trash size={13} />
      </button>
    </div>
  )

  const renderNode = (node: FileNode, depth: number) => {
    const pad = { paddingLeft: depth * 14 + 10 }

    if (editingId === node.path) {
      const stem = node.type === 'file' ? node.name.replace(/\.md$/i, '') : node.name
      return (
        <div key={node.path} style={pad} className="flex items-center py-[3px] pr-2">
          <span className="w-[18px] shrink-0" />
          <input
            autoFocus
            defaultValue={stem}
            onFocus={(event) => event.target.select()}
            onBlur={(event) => onRename(node.path, event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') event.currentTarget.blur()
              if (event.key === 'Escape') onRename(node.path, '')
            }}
            className="w-full min-w-0 rounded bg-base-800 px-1.5 py-0.5 text-[14px] text-text ring-1 ring-accent/40 outline-none"
          />
        </div>
      )
    }

    if (node.type === 'folder') {
      const isOpen = open.has(node.path)
      return (
        <div key={node.path}>
          <div className="group relative">
            <button
              draggable
              onClick={() => toggle(node.path)}
              onDragStart={(event) => {
                setDragPath(node.path)
                event.dataTransfer.effectAllowed = 'move'
                event.dataTransfer.setData('text/plain', node.path)
              }}
              onDragEnd={() => {
                setDragPath(null)
                setDropTarget(null)
              }}
              onDragOver={(event) => {
                if (!dragPath || dragPath === node.path) return
                if (node.path.startsWith(`${dragPath}/`)) return
                event.preventDefault()
                event.stopPropagation()
                setDropTarget(node.path)
              }}
              onDragLeave={() =>
                setDropTarget((current) => (current === node.path ? null : current))
              }
              onDrop={(event) => {
                event.preventDefault()
                event.stopPropagation()
                if (
                  dragPath &&
                  dragPath !== node.path &&
                  !node.path.startsWith(`${dragPath}/`)
                ) {
                  onMove(dragPath, node.path)
                }
                setDragPath(null)
                setDropTarget(null)
              }}
              style={pad}
              aria-expanded={isOpen}
              className={`flex w-full items-center gap-0.5 rounded-md py-[6px] pr-2 text-[14px] ${
                dropTarget === node.path
                  ? 'bg-accent-soft text-text ring-1 ring-accent/50'
                  : 'text-muted hover:bg-base-800 hover:text-text'
              }`}
            >
              <ChevronRight
                size={14}
                className={`shrink-0 transition-transform duration-150 ${isOpen ? 'rotate-90' : ''}`}
              />
              <span className="truncate">{node.name}</span>
            </button>
            {actions(node)}
          </div>
          {isOpen && node.children?.map((child) => renderNode(child, depth + 1))}
        </div>
      )
    }

    const isActive = node.path === activeId
    return (
      <div key={node.path} className="group relative">
        <button
          draggable
          onClick={() => onOpen(node.path)}
          onDragStart={(event) => {
            setDragPath(node.path)
            event.dataTransfer.effectAllowed = 'move'
            event.dataTransfer.setData('text/plain', node.path)
          }}
          onDragEnd={() => {
            setDragPath(null)
            setDropTarget(null)
          }}
          style={pad}
          aria-current={isActive ? 'page' : undefined}
          className={`flex w-full items-center rounded-md py-[6px] pr-2 text-[14px] ${
            isActive
              ? 'bg-accent-soft font-medium text-accent'
              : 'text-muted hover:bg-base-800 hover:text-text'
          }`}
        >
          <span className="w-[18px] shrink-0" />
          <span className="truncate">{node.name}</span>
        </button>
        {actions(node)}
      </div>
    )
  }

  return (
    <aside className="rise rise-1 flex h-full min-h-0 flex-col overflow-y-auto bg-base-900">
      {/* wordmark */}
      <div className="flex h-11 shrink-0 items-center border-b border-line px-4">
        <span className="text-[15px] font-semibold tracking-tight text-text">Local Quill</span>
      </div>

      {/* files */}
      <div className="px-3 pt-3">
        <div className="flex items-center justify-between px-1">
          <span className="text-[13px] font-semibold text-muted">Files</span>
          <div className="flex items-center gap-0.5">
            <button
              aria-label="New file"
              title="New file"
              onClick={onCreateFile}
              className="rounded p-1 text-muted hover:bg-base-800 hover:text-text"
            >
              <Plus size={15} />
            </button>
            <button
              aria-label="New folder"
              title="New folder"
              onClick={onCreateFolder}
              className="rounded p-1 text-muted hover:bg-base-800 hover:text-text"
            >
              <FolderPlus size={15} />
            </button>
          </div>
        </div>
        <nav
          className={`mt-1.5 space-y-px rounded-md ${
            dropTarget === '' ? 'ring-1 ring-accent/40' : ''
          }`}
          onDragOver={(event) => {
            if (!dragPath) return
            event.preventDefault()
            setDropTarget('')
          }}
          onDragLeave={() =>
            setDropTarget((current) => (current === '' ? null : current))
          }
          onDrop={(event) => {
            if (!dragPath) return
            event.preventDefault()
            if (dragPath.includes('/')) onMove(dragPath, '')
            setDragPath(null)
            setDropTarget(null)
          }}
        >
          {tree.length === 0 && (
            <p className="px-[9px] py-[6px] text-[12.5px] text-muted/50">
              No files yet. Create one to start writing.
            </p>
          )}
          {tree.map((node) => renderNode(node, 0))}
        </nav>
      </div>

      {/* reference library */}
      <div className="mt-4 border-t border-line px-3 pt-3 pb-4">
        <span className="px-1 text-[13px] font-semibold text-muted">Reference library</span>

        <div className="mt-1.5 space-y-px">
          <div className="flex items-center gap-2 rounded-md px-[9px] py-[6px] text-[14px] text-muted hover:bg-base-800 hover:text-text">
            <Bookmark size={14} className="shrink-0 opacity-60" />
            {outline ? (
              <>
                <span className="truncate" title={`${outline.chunks} chunks`}>
                  {outline.filename}
                </span>
                <button
                  aria-label="Clear outline"
                  title="Clear outline"
                  onClick={onClearOutline}
                  className="ml-auto shrink-0 rounded p-0.5 text-muted hover:bg-base-700 hover:text-text"
                >
                  <X size={12} />
                </button>
              </>
            ) : (
              <span className="text-muted/50">No outline set</span>
            )}
          </div>
          {documents.map((doc) => (
            <div
              key={doc.doc_id}
              title={`${doc.chunks} chunks`}
              className="flex items-center gap-2 rounded-md px-[9px] py-[6px] text-[14px] text-muted hover:bg-base-800 hover:text-text"
            >
              <FileText size={14} className="shrink-0 opacity-60" />
              <span className="truncate">{doc.filename}</span>
              <span className="ml-auto shrink-0 rounded-full border border-accent/40 px-1.5 py-0.5 text-[10.5px] text-accent">
                Processed
              </span>
            </div>
          ))}

          {uploads.map((entry) => (
            <div key={entry.id} className="rounded-md px-[9px] py-[6px] text-[14px]">
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
                <p className="mt-0.5 text-[11px] leading-snug text-mark/80">{entry.error}</p>
              )}
            </div>
          ))}
        </div>

        {documents.length === 0 && uploads.length === 0 && (
          <p className="mt-2 px-[9px] text-[12px] text-muted/50">
            No reference documents yet.
          </p>
        )}

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
          className={`mt-2 cursor-pointer rounded-lg border border-dashed px-3 py-4 text-center text-[12.5px] leading-snug transition-colors ${
            dragging
              ? 'border-accent/60 bg-accent-soft text-text'
              : 'border-line text-muted/70 hover:border-base-700 hover:text-muted'
          }`}
        >
          Drop documents here to use them as reference
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
      </div>
    </aside>
  )
}
