import type { FileNode } from './api'

export type { FileNode }

export function findNode(nodes: FileNode[], path: string): FileNode | undefined {
  for (const node of nodes) {
    if (node.path === path) return node
    const found = node.children ? findNode(node.children, path) : undefined
    if (found) return found
  }
  return undefined
}

export function collectFilePaths(node: FileNode): string[] {
  if (node.type === 'file') return [node.path]
  return (node.children ?? []).flatMap(collectFilePaths)
}

export function topLevelNames(nodes: FileNode[]): string[] {
  return nodes.map((node) => node.name)
}

export function uniqueName(name: string, existing: string[]): string {
  if (!existing.includes(name)) return name
  const dot = name.lastIndexOf('.')
  const stem = dot > 0 ? name.slice(0, dot) : name
  const extension = dot > 0 ? name.slice(dot) : ''
  let counter = 2
  while (existing.includes(`${stem} ${counter}${extension}`)) counter += 1
  return `${stem} ${counter}${extension}`
}
