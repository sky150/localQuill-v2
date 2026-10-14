export type FileNode = {
  name: string
  path: string
  type: 'file' | 'folder'
  children?: FileNode[]
}

export type DocumentInfo = {
  doc_id: string
  filename: string
  kind: string
  source: string
  chunks: number
  size_bytes: number
}

export type UploadResult = {
  doc_id: string
  filename: string
  kind: string
  added: number
  skipped: number
  total: number
}

export type TaskInfo = {
  model: string
  temperature: number
  collection: string | null
  top_k: number
}

export type ApiConfig = {
  styles: string[]
  tasks: Record<string, TaskInfo>
  upload_extensions: string[]
  upload_max_mb: number
  guardrails: boolean
}

export type ChatMessage = {
  role: 'user' | 'assistant'
  content: string
}

async function errorDetail(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: unknown }
    if (typeof body?.detail === 'string') return body.detail
  } catch {
    /* body was not json */
  }
  return `Request failed (${response.status})`
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, init)
  if (!response.ok) throw new Error(await errorDetail(response))
  return (await response.json()) as T
}

function json(method: string, body: unknown): RequestInit {
  return {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }
}

export const api = {
  getConfig: () => request<ApiConfig>('/config'),

  getTree: () => request<FileNode[]>('/files'),
  getContent: (path: string) =>
    request<{ path: string; content: string }>(
      `/files/content?path=${encodeURIComponent(path)}`,
    ),
  saveContent: (path: string, content: string) =>
    request<{ path: string }>('/files/content', json('PUT', { path, content })),
  createNode: (parent: string, name: string, kind: 'file' | 'folder') =>
    request<{ path: string }>('/files', json('POST', { parent, name, kind })),
  renameNode: (path: string, name: string) =>
    request<{ path: string }>('/files/rename', json('POST', { path, name })),
  moveNode: (path: string, parent: string) =>
    request<{ path: string }>('/files/move', json('POST', { path, parent })),
  deleteNode: (path: string) =>
    request<{ path: string }>(`/files?path=${encodeURIComponent(path)}`, {
      method: 'DELETE',
    }),

  getDocuments: (kind?: string) =>
    request<DocumentInfo[]>(
      kind ? `/documents?kind=${encodeURIComponent(kind)}` : '/documents',
    ),
  deleteDocument: (docId: string, deleteFile = false) =>
    request<{ doc_id: string }>(
      `/documents/${docId}?delete_file=${deleteFile}`,
      {
        method: 'DELETE',
      },
    ),
  indexWorkspaceDocument: (path: string, kind = 'outline') =>
    request<UploadResult>('/documents/from-workspace', json('POST', { path, kind })),

  chat: (payload: {
    task: string
    style: string
    text: string
    history: ChatMessage[]
    contextPaths: string[]
  }) =>
    request<{ result: string }>(
      '/chat',
      json('POST', {
        task: payload.task,
        style: payload.style,
        text: payload.text,
        history: payload.history,
        context_paths: payload.contextPaths,
      }),
    ),

  uploadDocument: (file: File, kind: string, onProgress: (percent: number) => void) =>
    new Promise<UploadResult>((resolve, reject) => {
      const xhr = new XMLHttpRequest()
      xhr.open('POST', `/api/documents?kind=${encodeURIComponent(kind)}`)

      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) {
          onProgress(Math.round((event.loaded / event.total) * 100))
        }
      }

      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(JSON.parse(xhr.responseText) as UploadResult)
          return
        }
        let message = `Upload failed (${xhr.status})`
        try {
          const body = JSON.parse(xhr.responseText) as { detail?: string }
          if (body.detail) message = body.detail
        } catch {
          /* body was not json */
        }
        reject(new Error(message))
      }

      xhr.onerror = () => reject(new Error('Upload failed'))

      const data = new FormData()
      data.append('file', file)
      xhr.send(data)
    }),
}
