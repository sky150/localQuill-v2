import { useState } from 'react'
import { ArrowUp, AtSign, FileText, X } from './icons'

const MODES = ['Q&A', 'Grammar', 'Feedback'] as const
type Mode = (typeof MODES)[number]

const CORRECTIONS = [
  {
    from: 'tuesday',
    to: 'Tuesday',
    why: 'Days of the week are capitalised.',
  },
  {
    from: 'her fathers study',
    to: "her father's study",
    why: 'Possession takes an apostrophe.',
  },
]

export default function Chat() {
  const [mode, setMode] = useState<Mode>('Grammar')

  return (
    <aside className="rise rise-3 flex h-full min-h-0 flex-col bg-base-900">
      {/* header */}
      <div className="flex h-11 shrink-0 items-center justify-between border-b border-line px-4">
        <span className="rounded-full border border-line px-2 py-0.5 text-[12px] text-muted">
          ministral-3:8b
        </span>
      </div>

      {/* conversation */}
      <div className="min-h-0 flex-1 space-y-6 overflow-y-auto p-4">
        <div className="ml-auto max-w-[88%] rounded-lg bg-base-800 px-3 py-2">
          <span className="inline-flex items-center gap-1.5 rounded-md bg-base-700/70 px-1.5 py-0.5 text-[12px] text-text">
            <FileText size={12} className="text-muted" />
            The Cartographer's Daughter.md
          </span>
          <p className="mt-1.5 text-[14px] text-text">Check the grammar on this page.</p>
        </div>

        <div>
          <div className="mb-1.5 text-[12px] font-medium text-muted">Quill</div>
          <p className="text-[14px] leading-relaxed text-text/90">
            Two things in the opening paragraph, both quick fixes:
          </p>

          <div className="mt-2.5 divide-y divide-line rounded-lg border border-line">
            {CORRECTIONS.map((c) => (
              <div key={c.from} className="px-3 py-2.5">
                <div className="flex flex-wrap items-baseline gap-x-2 text-[14px]">
                  <s className="text-mark/80 decoration-mark/60">{c.from}</s>
                  <span aria-hidden="true" className="text-muted">
                    →
                  </span>
                  <span className="font-medium text-text">{c.to}</span>
                </div>
                <p className="mt-1 text-[12.5px] leading-snug text-muted">{c.why}</p>
              </div>
            ))}
          </div>

          <p className="mt-2.5 text-[14px] leading-relaxed text-text/90">
            Everything else on this page is clean. The British spellings (&ldquo;colour&rdquo;,
            &ldquo;harbours&rdquo;) match your reference library, so I kept them.
          </p>
        </div>
      </div>

      {/* composer */}
      <div className="shrink-0 border-t border-line p-3">
        <div className="rounded-xl border border-line bg-base-850 transition-colors focus-within:border-accent/50">
          <div className="flex flex-wrap gap-1.5 px-2.5 pt-2.5">
            <span className="flex items-center gap-1.5 rounded-md bg-base-700/70 py-1 pr-1 pl-2 text-[12px] text-text">
              <FileText size={12} className="text-muted" />
              The Cartographer's Daughter.md
              <button
                aria-label="Remove context file"
                className="rounded p-0.5 text-muted hover:text-text"
              >
                <X size={12} />
              </button>
            </span>
          </div>

          <textarea
            rows={2}
            placeholder="Ask about this page…"
            className="mt-1 w-full resize-none bg-transparent px-3 py-1.5 text-[14px] text-text outline-none placeholder:text-muted/50"
          />

          <div className="flex items-center gap-1.5 px-2 pb-2">
            <button
              aria-label="Add context"
              title="Add context"
              className="flex size-7 shrink-0 items-center justify-center rounded-full border border-line text-muted hover:border-base-700 hover:text-text"
            >
              <AtSign size={13} />
            </button>
            <div
              className="flex min-w-0 rounded-full border border-line p-0.5"
              role="group"
              aria-label="Assistant mode"
            >
              {MODES.map((m) => (
                <button
                  key={m}
                  onClick={() => setMode(m)}
                  aria-pressed={mode === m}
                  className={`rounded-full px-2.5 py-1 text-[12px] whitespace-nowrap transition-colors ${
                    m === mode
                      ? 'bg-accent font-semibold text-base-950'
                      : 'text-muted hover:text-text'
                  }`}
                >
                  {m}
                </button>
              ))}
            </div>
            <button
              aria-label="Send"
              className="ml-auto flex size-7 shrink-0 items-center justify-center rounded-full bg-accent text-base-950 transition-colors hover:bg-accent-bright"
            >
              <ArrowUp size={14} />
            </button>
          </div>
        </div>
      </div>
    </aside>
  )
}
