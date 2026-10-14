import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { MouseEvent as ReactMouseEvent } from 'react'
import Chat, { type ChatTask } from './components/Chat'
import DocumentsModal from './components/DocumentsModal'
import Editor from './components/Editor'
import Sidebar, { type UploadEntry } from './components/Sidebar'
import { api, type ApiConfig, type DocumentInfo, type FileNode } from './lib/api'
import { collectFilePaths, findNode, topLevelNames, uniqueName } from './lib/tree'

let uploadCounter = 0

const SIDEBAR_DEFAULT = 280
const CHAT_DEFAULT = 360

function readWidth(key: string, fallback: number) {
  const value = Number(localStorage.getItem(key))
  return Number.isFinite(value) && value > 0 ? value : fallback
}

function ResizeHandle({
  onMouseDown,
  onDoubleClick,
}: {
  onMouseDown: (event: ReactMouseEvent) => void
  onDoubleClick: () => void
}) {
  return (
    <div
      role="separator"
      aria-orientation="vertical"
      onMouseDown={onMouseDown}
      onDoubleClick={onDoubleClick}
      className="w-[5px] shrink-0 cursor-col-resize bg-base-900 transition-colors hover:bg-accent/30"
    />
  )
}

export default function App() {
  const [tree, setTree] = useState<FileNode[]>([])
  const [tabs, setTabs] = useState<string[]>([])
  const [activePath, setActivePath] = useState<string | null>(null)
  const [contents, setContents] = useState<Record<string, string>>({})
  const [editingPath, setEditingPath] = useState<string | null>(null)
  const [documents, setDocuments] = useState<DocumentInfo[]>([])
  const [uploads, setUploads] = useState<UploadEntry[]>([])
  const [config, setConfig] = useState<ApiConfig | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [chatTask, setChatTask] = useState<ChatTask>('qa')
  const [attached, setAttached] = useState<string[]>([])
  const [documentsOpen, setDocumentsOpen] = useState(false)
  const [sidebarWidth, setSidebarWidth] = useState(() =>
    readWidth('lq.sidebar-width', SIDEBAR_DEFAULT),
  )
  const [chatWidth, setChatWidth] = useState(() =>
    readWidth('lq.chat-width', CHAT_DEFAULT),
  )
  const saveTimers = useRef<Record<string, ReturnType<typeof setTimeout>>>({})
  const widthsRef = useRef({ sidebar: sidebarWidth, chat: chatWidth })

  useEffect(() => {
    widthsRef.current = { sidebar: sidebarWidth, chat: chatWidth }
  }, [sidebarWidth, chatWidth])

  const startResize =
    (side: 'sidebar' | 'chat') => (event: ReactMouseEvent) => {
      event.preventDefault()
      const startX = event.clientX
      const startWidth = side === 'sidebar' ? sidebarWidth : chatWidth
      const setWidth = side === 'sidebar' ? setSidebarWidth : setChatWidth
      const min = side === 'sidebar' ? 200 : 300
      const max = side === 'sidebar' ? 480 : 620
      const direction = side === 'sidebar' ? 1 : -1
      document.body.style.userSelect = 'none'
      document.body.style.cursor = 'col-resize'
      const onMove = (move: MouseEvent) => {
        const delta = direction * (move.clientX - startX)
        setWidth(Math.min(max, Math.max(min, startWidth + delta)))
      }
      const onUp = () => {
        window.removeEventListener('mousemove', onMove)
        document.body.style.userSelect = ''
        document.body.style.cursor = ''
        localStorage.setItem('lq.sidebar-width', String(widthsRef.current.sidebar))
        localStorage.setItem('lq.chat-width', String(widthsRef.current.chat))
      }
      window.addEventListener('mousemove', onMove)
      window.addEventListener('mouseup', onUp, { once: true })
    }

  const resetWidth = (side: 'sidebar' | 'chat') => {
    if (side === 'sidebar') {
      setSidebarWidth(SIDEBAR_DEFAULT)
      localStorage.setItem('lq.sidebar-width', String(SIDEBAR_DEFAULT))
    } else {
      setChatWidth(CHAT_DEFAULT)
      localStorage.setItem('lq.chat-width', String(CHAT_DEFAULT))
    }
  }

  const refreshTree = useCallback(async () => {
    setTree(await api.getTree())
  }, [])

  const refreshDocuments = useCallback(async () => {
    setDocuments(await api.getDocuments())
  }, [])

  useEffect(() => {
    void refreshTree().catch((reason: Error) => setError(reason.message))
    void refreshDocuments().catch((reason: Error) => setError(reason.message))
    void api
      .getConfig()
      .then(setConfig)
      .catch((reason: Error) => setError(reason.message))
  }, [refreshTree, refreshDocuments])

  useEffect(() => {
    if (!activePath) return
    setAttached((current) =>
      current.includes(activePath) ? current : [...current, activePath],
    )
  }, [activePath])

  const openFile = async (path: string) => {
    setActivePath(path)
    setTabs((current) => (current.includes(path) ? current : [...current, path]))
    if (contents[path] === undefined) {
      try {
        const result = await api.getContent(path)
        setContents((current) => ({ ...current, [path]: result.content }))
      } catch (reason) {
        setError((reason as Error).message)
      }
    }
  }

  const closeTab = (path: string) => {
    const remaining = tabs.filter((tab) => tab !== path)
    setTabs(remaining)
    if (activePath === path) {
      setActivePath(remaining[remaining.length - 1] ?? null)
    }
  }

  const createNode = async (kind: 'file' | 'folder') => {
    const name =
      kind === 'file'
        ? uniqueName('Untitled.md', topLevelNames(tree))
        : uniqueName('New folder', topLevelNames(tree))
    try {
      const result = await api.createNode('', name, kind)
      await refreshTree()
      setEditingPath(result.path)
      if (kind === 'file') {
        setContents((current) => ({ ...current, [result.path]: '' }))
        setActivePath(result.path)
        setTabs((current) =>
          current.includes(result.path) ? current : [...current, result.path],
        )
      }
    } catch (reason) {
      setError((reason as Error).message)
    }
  }

  const deleteNode = async (path: string) => {
    const node = findNode(tree, path)
    if (!node) return
    const doomed = node.type === 'file' ? [path] : collectFilePaths(node)
    try {
      await api.deleteNode(path)
      setTabs((current) => current.filter((tab) => !doomed.includes(tab)))
      setContents((current) => {
        const next = { ...current }
        doomed.forEach((filePath) => delete next[filePath])
        return next
      })
      if (activePath && doomed.includes(activePath)) setActivePath(null)
      setAttached((current) => current.filter((entry) => !doomed.includes(entry)))
      if (editingPath === path) setEditingPath(null)
      await refreshTree()
    } catch (reason) {
      setError((reason as Error).message)
    }
  }

  const applyPathChange = (path: string, newPath: string) => {
    const remap = (candidate: string) =>
      candidate === path
        ? newPath
        : candidate.startsWith(`${path}/`)
          ? `${newPath}${candidate.slice(path.length)}`
          : candidate

    setTabs((current) => current.map(remap))
    setContents((current) => {
      const next: Record<string, string> = {}
      for (const [key, content] of Object.entries(current)) {
        next[remap(key)] = content
      }
      return next
    })
    setActivePath((current) => (current ? remap(current) : current))
    setAttached((current) => current.map(remap))
  }

  const commitRename = async (path: string, raw: string) => {
    setEditingPath(null)
    const node = findNode(tree, path)
    const value = raw.trim()
    if (!node || !value) return
    const name =
      node.type === 'file' && !value.toLowerCase().endsWith('.md')
        ? `${value}.md`
        : value
    if (name === node.name) return
    try {
      const result = await api.renameNode(path, name)
      applyPathChange(path, result.path)
      await refreshTree()
    } catch (reason) {
      setError((reason as Error).message)
    }
  }

  const moveNode = async (path: string, parent: string) => {
    try {
      const result = await api.moveNode(path, parent)
      applyPathChange(path, result.path)
      await refreshTree()
    } catch (reason) {
      setError((reason as Error).message)
    }
  }

  const handleChange = (value: string) => {
    if (!activePath) return
    const path = activePath
    setContents((current) => ({ ...current, [path]: value }))
    clearTimeout(saveTimers.current[path])
    saveTimers.current[path] = setTimeout(() => {
      api.saveContent(path, value).catch((reason: Error) => setError(reason.message))
    }, 800)
  }

  const uploadFiles = (files: File[]) => {
    for (const file of files) {
      const id = `upload-${++uploadCounter}`
      setUploads((current) => [
        ...current,
        { id, name: file.name, progress: 0, status: 'uploading' },
      ])
      api
        .uploadDocument(file, 'userdocs', (progress) => {
          setUploads((current) =>
            current.map((entry) =>
              entry.id === id
                ? {
                    ...entry,
                    progress,
                    status: progress >= 100 ? 'processing' : 'uploading',
                  }
                : entry,
            ),
          )
        })
        .then(async () => {
          await refreshDocuments()
          setUploads((current) => current.filter((entry) => entry.id !== id))
        })
        .catch((reason: Error) => {
          setUploads((current) =>
            current.map((entry) =>
              entry.id === id
                ? { ...entry, status: 'error', error: reason.message }
                : entry,
            ),
          )
        })
    }
  }

  const openFiles = tabs.map((path) => ({
    id: path,
    name: findNode(tree, path)?.name ?? path.split('/').pop() ?? path,
  }))
  const content = activePath ? (contents[activePath] ?? '') : ''
  const wordCount = useMemo(
    () => content.trim().split(/\s+/).filter(Boolean).length,
    [content],
  )

  const clearOutlineDocs = async () => {
    const existing = documents.filter((doc) => doc.kind === 'outline')
    await Promise.all(existing.map((doc) => api.deleteDocument(doc.doc_id)))
  }

  const setOutline = async (path: string) => {
    try {
      await clearOutlineDocs()
      await api.indexWorkspaceDocument(path, 'outline')
      await refreshDocuments()
    } catch (reason) {
      setError((reason as Error).message)
    }
  }

  const clearOutline = async () => {
    try {
      await clearOutlineDocs()
      await refreshDocuments()
    } catch (reason) {
      setError((reason as Error).message)
    }
  }

  const removeDocument = async (docId: string) => {
    try {
      await api.deleteDocument(docId, true)
      await refreshDocuments()
    } catch (reason) {
      setError((reason as Error).message)
    }
  }

  const userdocs = documents.filter((doc) => doc.kind === 'userdocs')
  const outline = documents.find((doc) => doc.kind === 'outline') ?? null
  const contextFiles = useMemo(
    () =>
      tree
        .flatMap(collectFilePaths)
        .filter((path) => /\.(md|markdown|txt)$/i.test(path)),
    [tree],
  )
  const outlinePath = useMemo(() => {
    if (!outline) return null
    return (
      contextFiles.find((path) => outline.source.endsWith(`/${path}`)) ?? null
    )
  }, [outline, contextFiles])

  return (
    <div className="flex h-screen bg-base-950">
      <div className="shrink-0" style={{ width: sidebarWidth }}>
        <Sidebar
          tree={tree}
          documents={userdocs}
          outline={outline}
          outlinePath={outlinePath}
          uploadsActive={uploads.some((entry) => entry.status !== 'error')}
          activeId={activePath}
          editingId={editingPath}
          onOpen={openFile}
          onCreateFile={() => void createNode('file')}
          onCreateFolder={() => void createNode('folder')}
          onDelete={deleteNode}
          onRename={commitRename}
          onMove={moveNode}
          onClearOutline={() => void clearOutline()}
          onManageDocuments={() => setDocumentsOpen(true)}
        />
      </div>
      <ResizeHandle
        onMouseDown={startResize('sidebar')}
        onDoubleClick={() => resetWidth('sidebar')}
      />
      <Editor
        openFiles={openFiles}
        activeId={activePath}
        content={content}
        wordCount={wordCount}
        onSelect={openFile}
        onClose={closeTab}
        onChange={handleChange}
      />
      <ResizeHandle
        onMouseDown={startResize('chat')}
        onDoubleClick={() => resetWidth('chat')}
      />
      <div className="shrink-0" style={{ width: chatWidth }}>
        <Chat
          config={config}
          files={contextFiles}
          task={chatTask}
          onTaskChange={setChatTask}
          attached={attached}
          onAttach={(path) =>
            setAttached((current) =>
              current.includes(path) ? current : [...current, path],
            )
          }
          onDetach={(path) =>
            setAttached((current) => current.filter((entry) => entry !== path))
          }
        />
      </div>
      <DocumentsModal
        open={documentsOpen}
        documents={userdocs}
        outline={outline}
        outlinePath={outlinePath}
        uploads={uploads}
        acceptExtensions={config?.upload_extensions ?? []}
        files={contextFiles}
        onClose={() => setDocumentsOpen(false)}
        onUpload={uploadFiles}
        onDeleteDocument={(docId) => void removeDocument(docId)}
        onDismissUpload={(id) =>
          setUploads((current) => current.filter((entry) => entry.id !== id))
        }
        onSelectOutline={(path) => void setOutline(path)}
        onClearOutline={() => void clearOutline()}
      />
      {error && (
        <button
          onClick={() => setError(null)}
          className="fixed bottom-4 left-1/2 z-50 max-w-[80%] -translate-x-1/2 rounded-lg border border-mark/40 bg-base-800 px-3 py-2 text-[12.5px] text-mark shadow-lg"
        >
          {error}
        </button>
      )}
    </div>
  )
}
