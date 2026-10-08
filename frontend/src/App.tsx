import { useMemo, useState } from 'react'
import Chat from './components/Chat'
import Editor from './components/Editor'
import Sidebar from './components/Sidebar'
import {
  collectFileIds,
  findNode,
  removeNode,
  renameNode,
  uniqueName,
  type FileNode,
} from './lib/tree'

const CARTOGRAPHER_ID = 'f-cartographer'

const CARTOGRAPHER_TEXT = `Pell found the map on a tuesday morning, folded between the floorboards of her fathers study. The ink had faded to the colour of weak tea, and the coastline it described matched no shore she knew.

She traced the route with one finger. Three harbours, a reef drawn with teeth, and in her mother's careful hand a single sentence along the margin: Do not let the Admiralty see this.

Below it, in newer ink, her own hand: Then we sail before winter.`

const initialTree: FileNode[] = [
  {
    id: 'd-manuscripts',
    name: 'Manuscripts',
    type: 'folder',
    children: [
      { id: CARTOGRAPHER_ID, name: "The Cartographer's Daughter.md", type: 'file' },
      { id: 'f-bell', name: 'Bell Tower.md', type: 'file' },
    ],
  },
  {
    id: 'd-formal',
    name: 'Formal',
    type: 'folder',
    children: [
      { id: 'f-cover', name: 'Cover Letter.md', type: 'file' },
      { id: 'f-motivation', name: 'Motivation Letter.md', type: 'file' },
    ],
  },
  { id: 'f-outline', name: 'Outline.md', type: 'file' },
  { id: 'f-research', name: 'Research Notes.md', type: 'file' },
]

const initialContents: Record<string, string> = {
  [CARTOGRAPHER_ID]: CARTOGRAPHER_TEXT,
  'f-bell': 'The bell had not rung in forty years, but Pell swore she heard it the night the harbour froze.',
  'f-cover': 'Dear Mr. Okafor,',
  'f-motivation': '',
  'f-outline': '',
  'f-research': '',
}

export default function App() {
  const [tree, setTree] = useState<FileNode[]>(initialTree)
  const [tabs, setTabs] = useState<string[]>([CARTOGRAPHER_ID])
  const [activeId, setActiveId] = useState<string | null>(CARTOGRAPHER_ID)
  const [contents, setContents] = useState<Record<string, string>>(initialContents)
  const [editingId, setEditingId] = useState<string | null>(null)

  const openFile = (id: string) => {
    setActiveId(id)
    setTabs((current) => (current.includes(id) ? current : [...current, id]))
  }

  const closeTab = (id: string) => {
    const remaining = tabs.filter((tab) => tab !== id)
    setTabs(remaining)
    if (activeId === id) {
      setActiveId(remaining[remaining.length - 1] ?? null)
    }
  }

  const createFile = () => {
    const id = crypto.randomUUID()
    const name = uniqueName('Untitled.md', tree.map((node) => node.name))
    setTree((current) => [...current, { id, name, type: 'file' }])
    setContents((current) => ({ ...current, [id]: '' }))
    setEditingId(id)
    openFile(id)
  }

  const createFolder = () => {
    const id = crypto.randomUUID()
    const name = uniqueName('New folder', tree.map((node) => node.name))
    setTree((current) => [...current, { id, name, type: 'folder', children: [] }])
    setEditingId(id)
  }

  const deleteNode = (id: string) => {
    const node = findNode(tree, id)
    if (!node) return
    const doomed = node.type === 'file' ? [id] : collectFileIds(node)

    setTree((current) => removeNode(current, id))
    setContents((current) => {
      const next = { ...current }
      doomed.forEach((fileId) => delete next[fileId])
      return next
    })

    const remaining = tabs.filter((tab) => !doomed.includes(tab))
    setTabs(remaining)
    if (activeId && doomed.includes(activeId)) {
      setActiveId(remaining[remaining.length - 1] ?? null)
    }
    if (editingId === id) {
      setEditingId(null)
    }
  }

  const commitRename = (id: string, raw: string) => {
    setEditingId(null)
    const node = findNode(tree, id)
    const value = raw.trim()
    if (!node || !value) return
    const name =
      node.type === 'file' && !value.toLowerCase().endsWith('.md') ? `${value}.md` : value
    setTree((current) => renameNode(current, id, name))
  }

  const openFiles = tabs.map((id) => ({ id, name: findNode(tree, id)?.name ?? 'Untitled' }))
  const content = activeId ? (contents[activeId] ?? '') : ''
  const wordCount = useMemo(
    () => content.trim().split(/\s+/).filter(Boolean).length,
    [content],
  )

  return (
    <div className="grid h-screen grid-cols-[280px_minmax(0,1fr)_360px] bg-base-950">
      <Sidebar
        tree={tree}
        activeId={activeId}
        editingId={editingId}
        onOpen={openFile}
        onCreateFile={createFile}
        onCreateFolder={createFolder}
        onDelete={deleteNode}
        onRename={commitRename}
      />
      <Editor
        openFiles={openFiles}
        activeId={activeId}
        content={content}
        wordCount={wordCount}
        onSelect={openFile}
        onClose={closeTab}
        onChange={(value) =>
          setContents((current) => (activeId ? { ...current, [activeId]: value } : current))
        }
      />
      <Chat />
    </div>
  )
}
