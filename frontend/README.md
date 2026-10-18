# Local Quill — Frontend

React + TypeScript + Vite single-page app for Local Quill. It talks to the FastAPI backend through a dev proxy: requests to `/api/...` are forwarded to `http://localhost:8000` (configured in `vite.config.ts`), so no CORS setup is needed.

## Prerequisites

- **Node.js 22 LTS or newer** (Vite 8 requires Node 20.19+). Check with `node -v` and `npm -v`.
- The **Local Quill API** running on port 8000 (from the repository root).
- **Ollama** running with the models your tasks use (needed for chat; the file editor works without it).

Install Node with [nvm](https://github.com/nvm-sh/nvm) (macOS/Linux), [nvm-windows](https://github.com/coreybutler/nvm-windows), [fnm](https://github.com/Schniz/fnm), or the [official installer](https://nodejs.org).

## Install

```sh
cd .\frontend\
npm install
```

The same command works on Windows, Linux and macOS.

## Run (development)

1. Start Ollama (if it is not already running as a background app):

   ```sh
   ollama serve
   ```

2. Start the API from the repository root.

   macOS / Linux:

   ```sh
   uv run uvicorn src.api.main:app --reload --port 8000
   ```

   Windows (PowerShell):

   ```powershell
   $env:PYTHONPATH="."; uv run uvicorn src.api.main:app --reload --port 8000
   ```

3. Start the frontend from this folder:

   ```sh
   cd \.frontend\
   npm run dev
   ```

4. Open `http://localhost:5173`.

The Vite dev server proxies `/api` to `http://localhost:8000`, so leave the API on port 8000 unless you also update the proxy target.

## Scripts

| Command | Description |
| --- | --- |
| `npm run dev` | Start the dev server with hot reload |
| `npm run build` | Type-check and build the production bundle into `dist/` |
| `npm run preview` | Serve the production build locally |
| `npm run lint` | Run oxlint |

## How data flows

- **Workspace files** (the editor sidebar) are stored by the API under `WORKSPACE_PATH` (default `./data/workspace`).
- **Reference library** uploads are stored under `UPLOAD_PATH` (default `./data/uploads`) and indexed into ChromaDB.
- **Vectors** live under `CHROMA_PATH` (default `./chroma`).
- Paths and model settings come from the `.env` file at the repository root.

If the API is not running, the file tree will be empty and an error toast appears at the bottom.

## Troubleshooting

- **Port 5173 already in use** — run `npm run dev -- --port 5174`; the proxy keeps working.
- **Port 8000 already in use** — stop the other process or run uvicorn on another port and update the proxy target in `vite.config.ts`.
- **Chat errors / "LLM unavailable"** — Ollama is not running or the model is not pulled; check with `ollama list`.
- **Windows firewall prompt on first run** — allow Node.js so the browser can reach the dev server.

## Project structure

```
src/
  components/    Sidebar, Editor, Chat, Documents modal, icons
  lib/api.ts     Typed API client (also XHR upload with progress)
  lib/tree.ts    File tree helpers
  App.tsx        App state: files, tabs, attachments, uploads
```
