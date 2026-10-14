import { useState } from 'react'
import type { DocumentInfo, FileNode } from '../lib/api'
import { Bookmark, ChevronRight, FileText, FolderPlus, Plus, Trash, Upload, X } from './icons'

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
  outlinePath: string | null
  uploadsActive: boolean
  activeId: string | null
  editingId: string | null
  onOpen: (path: string) => void
  onCreateFile: () => void
  onCreateFolder: () => void
  onDelete: (path: string) => void
  onRename: (path: string, value: string) => void
  onMove: (path: string, parent: string) => void
  onClearOutline: () => void
  onManageDocuments: () => void
}

export default function Sidebar({
  tree,
  documents,
  outline,
  outlinePath,
  uploadsActive,
  activeId,
  editingId,
  onOpen,
  onCreateFile,
  onCreateFolder,
  onDelete,
  onRename,
  onMove,
  onClearOutline,
  onManageDocuments,
}: SidebarProps) {
  const [open, setOpen] = useState<ReadonlySet<string>>(new Set())
  const [dragPath, setDragPath] = useState<string | null>(null)
  const [dropTarget, setDropTarget] = useState<string | null>(null)

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

  const actions = (node: FileNode) => (
    <div className="absolute top-1/2 right-2 flex -translate-y-1/2 items-center gap-0.5 opacity-0 transition-opacity group-hover:opacity-100">
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
    const pad = { paddingLeft: depth * 14 + 8 }

    if (editingId === node.path) {
      const stem = node.type === 'file' ? node.name.replace(/\.md$/i, '') : node.name
      return (
        <div key={node.path} style={pad} className="flex items-center py-[3px] pr-2">
          <span className="w-[22px] shrink-0" />
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
    const isOutline = node.path === outlinePath
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
          <span className="w-[22px] shrink-0" />
          <span className="truncate">{node.name}</span>
          {isOutline && (
            <Bookmark
              size={12}
              className="ml-auto mr-1 shrink-0 text-accent group-hover:opacity-0"
              aria-label="Current outline"
            />
          )}
        </button>
        {actions(node)}
      </div>
    )
  }

  return (
    <aside className="rise rise-1 flex h-full min-h-0 flex-col overflow-y-auto bg-base-900">
      {/* wordmark */}
      <div className="flex h-11 shrink-0 items-center border-b border-line px-4">
        <span className="text-[20px] font-semibold tracking-tight text-text">Local Quill</span>
      </div>

      {/* files */}
      <div className="px-3 pt-3">
        <div className="flex items-center justify-between px-2">
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
            <p className="px-2 py-[6px] text-[12.5px] text-muted/50">
              No files yet. Create one to start writing.
            </p>
          )}
          {tree.map((node) => renderNode(node, 0))}
        </nav>
      </div>

      {/* outline */}
      <div className="mt-4 border-t border-line px-3 pt-3">
        <span className="px-2 text-[13px] font-semibold text-muted">Outline</span>
        <div className="mt-1.5">
          {outline ? (
            <div className="group flex items-center gap-2 rounded-md px-2 py-[6px] text-[14px] text-muted hover:bg-base-800 hover:text-text">
              <Bookmark size={14} className="shrink-0 text-accent" />
              <button
                onClick={() => outlinePath && onOpen(outlinePath)}
                title={`${outline.chunks} chunks${outlinePath ? ' — click to open' : ''}`}
                className="truncate text-left"
              >
                {outline.filename}
              </button>
              <button
                aria-label="Clear outline"
                title="Clear outline"
                onClick={onClearOutline}
                className="ml-auto shrink-0 rounded p-0.5 text-muted opacity-0 transition-opacity group-hover:opacity-100 hover:bg-base-700 hover:text-text"
              >
                <X size={12} />
              </button>
            </div>
          ) : (
            <p className="px-2 py-[6px] text-[12.5px] text-muted/50">
              No outline set.
            </p>
          )}
        </div>
      </div>

      {/* reference library */}
      <div className="mt-4 border-t border-line px-3 pt-3 pb-4">
        <span className="px-2 text-[13px] font-semibold text-muted">Reference library</span>

        <div className="mt-1.5 space-y-px">
          {documents.map((doc) => (
            <div
              key={doc.doc_id}
              title={`${doc.chunks} chunks`}
              className="flex items-center gap-2 rounded-md px-2 py-[6px] text-[14px] text-muted hover:bg-base-800 hover:text-text"
            >
              <FileText size={14} className="shrink-0 opacity-60" />
              <span className="truncate">{doc.filename}</span>
            </div>
          ))}
        </div>

        {documents.length === 0 && (
          <p className="mt-1 px-2 text-[12px] text-muted/50">
            No reference documents yet.
          </p>
        )}

        <button
          onClick={onManageDocuments}
          className="mt-2 flex w-full items-center justify-center gap-1.5 rounded-lg border border-dashed border-line px-3 py-2.5 text-[12.5px] text-muted/70 transition-colors hover:border-base-700 hover:text-muted"
        >
          <Upload size={13} className={uploadsActive ? 'animate-pulse text-accent' : ''} />
          {uploadsActive ? 'Uploading…' : 'Add or manage documents'}
        </button>
      </div>
    </aside>
  )
}
