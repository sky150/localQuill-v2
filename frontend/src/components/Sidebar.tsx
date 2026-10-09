import { useState } from 'react'
import type { FileNode } from '../lib/tree'
import { ChevronRight, FileText, FolderPlus, Plus, Trash } from './icons'

const LIBRARY = ['British Style Guide.pdf', 'Chicago Manual of Style.pdf']

type SidebarProps = {
  tree: FileNode[]
  activeId: string | null
  editingId: string | null
  onOpen: (id: string) => void
  onCreateFile: () => void
  onCreateFolder: () => void
  onDelete: (id: string) => void
  onRename: (id: string, value: string) => void
}

export default function Sidebar({
  tree,
  activeId,
  editingId,
  onOpen,
  onCreateFile,
  onCreateFolder,
  onDelete,
  onRename,
}: SidebarProps) {
  const [open, setOpen] = useState<ReadonlySet<string>>(new Set(['d-manuscripts']))

  const toggle = (id: string) =>
    setOpen((prev) => {
      const next = new Set(prev)
      if (next.has(id)) {
        next.delete(id)
      } else {
        next.add(id)
      }
      return next
    })

  const deleteButton = (node: FileNode) => (
    <button
      aria-label={`Delete ${node.name}`}
      title={`Delete ${node.name}`}
      onClick={(event) => {
        event.stopPropagation()
        onDelete(node.id)
      }}
      className="absolute top-1/2 right-1 -translate-y-1/2 rounded bg-base-800 p-1 text-muted opacity-0 transition-opacity group-hover:opacity-100 hover:bg-base-700 hover:text-text"
    >
      <Trash size={13} />
    </button>
  )

  const renderNode = (node: FileNode, depth: number) => {
    const pad = { paddingLeft: depth * 14 + 10 }

    if (editingId === node.id) {
      const stem = node.type === 'file' ? node.name.replace(/\.md$/i, '') : node.name
      return (
        <div key={node.id} style={pad} className="flex items-center py-[3px] pr-2">
          <span className="w-[18px] shrink-0" />
          <input
            autoFocus
            defaultValue={stem}
            onFocus={(event) => event.target.select()}
            onBlur={(event) => onRename(node.id, event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') event.currentTarget.blur()
              if (event.key === 'Escape') onRename(node.id, '')
            }}
            className="w-full min-w-0 rounded bg-base-800 px-1.5 py-0.5 text-[14px] text-text ring-1 ring-accent/40 outline-none"
          />
        </div>
      )
    }

    if (node.type === 'folder') {
      const isOpen = open.has(node.id)
      return (
        <div key={node.id}>
          <div className="group relative">
            <button
              onClick={() => toggle(node.id)}
              style={pad}
              aria-expanded={isOpen}
              className="flex w-full items-center gap-0.5 rounded-md py-[6px] pr-2 text-[14px] text-muted hover:bg-base-800 hover:text-text"
            >
              <ChevronRight
                size={14}
                className={`shrink-0 transition-transform duration-150 ${isOpen ? 'rotate-90' : ''}`}
              />
              <span className="truncate">{node.name}</span>
            </button>
            {deleteButton(node)}
          </div>
          {isOpen && node.children?.map((child) => renderNode(child, depth + 1))}
        </div>
      )
    }

    const isActive = node.id === activeId
    return (
      <div key={node.id} className="group relative">
        <button
          onClick={() => onOpen(node.id)}
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
        {deleteButton(node)}
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
        <nav className="mt-1.5 space-y-px">{tree.map((node) => renderNode(node, 0))}</nav>
      </div>

      {/* personal reference library — separate section, wired up later */}
      <div className="mt-4 border-t border-line px-3 pt-3 pb-4">
        <span className="px-1 text-[13px] font-semibold text-muted">Reference library</span>
        <div className="mt-1.5 space-y-px">
          {LIBRARY.map((name) => (
            <div
              key={name}
              className="flex items-center gap-2 rounded-md px-[9px] py-[6px] text-[14px] text-muted hover:bg-base-800 hover:text-text"
            >
              <FileText size={14} className="shrink-0 opacity-60" />
              <span className="truncate">{name}</span>
            </div>
          ))}
        </div>
        <div className="mt-2 rounded-lg border border-dashed border-line px-3 py-4 text-center text-[12.5px] leading-snug text-muted/70">
          Drop PDFs or .md files here to use them as reference
        </div>
      </div>
    </aside>
  )
}
