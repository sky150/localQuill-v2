# RAG evaluation roadmap

Goal: find the best retrieval setup for each AI task (grammar, feedback, Q&A), then the best
embedding model for the app.

## Where we are

### What we did before (March–May 2026)

`tests/eval/run_retrieval_eval.py`, results in `evaluation/reports/DISPRO1/eval_retrieval_results.jsonl`
(37 runs):

| Tested | Values |
|---|---|
| Embedding models | nomic-embed-text, mxbai-embed-large, Qwen3-Embedding-0.6B, embeddinggemma, all-MiniLM |
| Chunk size / overlap | 500/100, 1000/200, 1000/400 |
| top_k | 3 vs 5 (nomic only) |
| Metrics | deepeval contextual precision/recall (judge: llama3.1:8b), document hit rate |
| Data | 4 questions on the formal style guides |

**Why it couldn't pick a winner:**
- Hit rate was 1.0 in every run, so it didn't separate anything.
- The judge's noise was as large as the gaps between models: mxbai with identical settings
  scored recall 0.58, 0.63, 0.67 and 0.68 across four runs.
- One question type only, and not how the app retrieves today.

The current `.env` setting (chunk 1000, overlap 400) comes from those runs.

### How the tasks retrieved before this evaluation

| Task | Collection(s) | Query | top_k |
|---|---|---|---|
| Grammar | Style guide (`formal` / `fiction`) | Fixed keyword list + first 2,000 chars of the draft | 3 |
| Feedback | Style guide + `outline` | Keywords + draft; draft only for the outline | 2 + 1 |
| Q&A | `userdocs` + `outline` | The user's question | 3 + 1 |

Chunk size is set **per collection** (all tasks reading it share it). top_k and the query are
set **per task**.

### Issues found while preparing

1. **The formal collection has almost no grammar rules.** Comma splices, run-ons and
   subject-verb agreement only appear in *Writing for Success*, which is in the `fiction`
   collection. Grammar in formal style retrieves from a collection with nothing to find.
2. **The draft dominates the grammar/feedback query.** 2,000 characters of draft vs. one line
   of keywords, so the search likely returns guide chunks about the essay's *topic* rather
   than writing advice.
3. **No query/document prefixes.** nomic (`search_query:` / `search_document:`),
   Qwen3-Embedding (query instruction) and EmbeddingGemma (task prompts) expect them;
   `OllamaEmbeddings` sends none, so these models are tested below their potential.
4. **nomic's context on Ollama defaults to 2,048 tokens.**

## Phases

### Phase 1: Test sets (done, reviewed)

In `tests/eval/rag/`. Answers are stored as **text snippets**, not chunk IDs, so the same set
works for any chunk size: a retrieved chunk is a hit if it contains the snippet.

| File | Cases | Corpus |
|---|---|---|
| `testset_qa.json` | 30 questions (18 about a novel, 8 about essays, 3 about an outline, 1 mixed) | *The Wonderful Wizard of Oz* + 4 C2 responses as `userdocs`; `corpus/outline/Outline_Technology_Essay.md` as `outline` |
| `testset_style.json` | 20 drafts (10 formal, 10 fiction) with one known problem each | The `formal` and `fiction` style guides |

`validate_testsets.py` checks that every snippet exists in the text the app indexes (all 56 do).
Each snippet must contain the answer or rule itself, not just a nearby line or a heading. In
Q&A, a case with several snippets needs all of them for a complete answer; in the style set,
any one of them is useful on its own.

**Review checklist (human):**
- [ ] Each Q&A snippet actually answers its question.
- [ ] Each style snippet is advice a writer would need for that draft.
- [ ] No obviously better passage is missing from `gold` (add it if there is one).
- [ ] Questions sound like something a real user would ask.

### Phase 2: Eval harness (done)

`tests/eval/rag/run_rag_eval.py` (run it, then `summarize_rag_eval.py` for tables):
- Builds a separate Chroma index per (model, chunk size, overlap), cached so it is built once.
- Runs both test sets and builds queries the way each task does.
- Metrics, no LLM judge:
  - **Hit@k**: at least one gold snippet in the top k.
  - **Recall@k**: share of gold snippets found in the top k (for Q&A: 1.0 = complete answer).
  - **MRR**: 1 / rank of the first hit (rewards putting the answer first).
- Also records index time, query latency and model size.
- Writes one JSON line per run to `evaluation/reports/RAG/`; charts in a notebook.

Embeddings are deterministic, so one run per setup is enough (no repeated runs).

### Phase 3: Best configuration per task (done for nomic)

Fixed model (nomic as baseline, plus one strong model to check results hold).

| Parameter | Values | Applies to |
|---|---|---|
| Chunk size | 500, 1000, 1500 | Per collection |
| Overlap | 0, 200, 400 | Per collection |
| top_k | 1–8 | Per task |
| Query style | keywords only, draft only, keywords + draft, LLM-written query | Grammar, feedback |
| Prefixes | on / off | Per model |

Also test whether grammar in formal style improves when it may also search *Writing for Success*
(issue 1).

### Phase 4: Embedding model comparison (done)

With the best configuration per task. All available on Ollama (checked November 2026):

| Model | Ollama tag | Notes |
|---|---|---|
| nomic-embed-text | `nomic-embed-text` | Current setting, baseline |
| EmbeddingGemma | `embeddinggemma` | 300m, Gemma 3; tested before |
| EmbeddingGemma 2 | `embeddinggemma-2:270m`, `:740m` | Gemma 4-based (Oct 2026). On Ollama `:740m` is the same 271M text model plus a vision/audio projector, so for text both give identical results |
| Qwen3-Embedding | `qwen3-embedding:0.6b` (`:4b` if fast enough) | Strong on benchmarks, long context |
| mxbai-embed-large | `mxbai-embed-large` | Tested before, for comparison |

Optional: `granite-embedding` (small, IBM). Dropped: all-MiniLM (old, short context).

Pick on retrieval quality first, then speed and memory (it runs on the user's machine).
Changing the model means re-indexing every collection, including users' uploads.

### Phase 5: End-to-end check (done)

`tests/eval/rag/run_task_eval.py` runs grammar and feedback on every draft in `testset_style.json`
with different guideline variants (fixed file, style-guide search, both, none, right passages) and
asks a judge whether the known problem was fixed (grammar) or named (feedback); feedback is graded
by hand. Results below under Findings 4 to 6.

### Phase 6: Apply and document

- Set chunk size per collection, and top_k and query style per task (`src/config.py`, `.env`).
- Switch the embedding model and re-index.
- Write up the results (charts from Phase 2–5) for the report and presentation.

## Results (November 2026)

Charts and tables (they read the files in `evaluation/reports/`, re-run nothing):
- `evaluation/notebooks/8_rag_evaluation.ipynb`: test sets, metrics (and why DeepEval was dropped),
  retrieval config, prompt/config changes before vs. after, Phase 5 results, all decisions.
- `evaluation/notebooks/9_embedding_model_decision.ipynb`: the embedding model comparison.

Raw runs: `evaluation/reports/RAG/results.jsonl` (38 runs), console output in
`phase3_nomic.txt` and `phase4_models.txt`. Tables: `uv run python tests/eval/rag/summarize_rag_eval.py`.

**Read with care:** the sets are small (Q&A 26 userdocs questions, grammar 7, feedback 13 cases), so
one case moves Q&A by 0.04, grammar by 0.14 and feedback by 0.08. Only patterns that repeat across
many setups are reported as findings.

### Finding 1: the style query is the problem, not chunk size or model

Grammar Hit@3 and feedback Hit@2, mean over the 18 nomic setups (chunk 500/1000/1500 x overlap
0/200/400 x prefixes off/on):

| Query | Grammar Hit@3 | Feedback Hit@2 |
|---|---|---|
| keywords + text (`retrieval_query` + `text[:2000]`, the app before) | 0.32 | 0.01 |
| text only (`text[:2000]`) | 0.05 | 0.00 |
| keywords only (`retrieval_query`) | 0.35 | 0.02 |
| query written by the model (≤12 words, `ministral-3:8b`) | 0.38 | 0.15 |
| test set answer (the `issue` field; best case, the app can't know it) | 0.77 | 0.58 |
| label picked by the model + re-ranked (app now, Finding 6; all 20 texts, chunk 1000/400 only) | 0.67 | 0.64 |

With the draft in the query, the search returns guide passages about the draft's *topic* (a
lighthouse story gets passages about boats and the sea; an essay about policy costs gets grant
proposal budgets). Feedback today almost never gets the relevant advice. Every model shows the same
pattern. A query that names the problem (test set answer) works far better; a query written by
the model gets part of the way. `qwen3.5:2b` was too small to write the query (it repeated the prompt), so this needs a
4B+ model and costs one extra LLM call per request.

### Finding 2: Q&A works best with no overlap

Q&A Hit@3 at chunk 1000, prefixes off:

| Model | Overlap 0 | Overlap 200 | Overlap 400 (current) |
|---|---|---|---|
| nomic-embed-text | **0.88** | 0.62 | 0.69 |
| embeddinggemma | 0.85 | 0.73 | – |
| embeddinggemma-2 | 0.81 | 0.77 | – |
| qwen3-embedding:0.6b | 0.73 | 0.50 | – |
| mxbai-embed-large | 0.81 | 0.73 | – |

Overlap 0 wins for every model. With large overlap, neighbouring chunks are near-duplicates and
crowd the top 3 with the same passage. Chunk 1000 beat 500 and 1500 for nomic.

### Finding 3: no clear best embedding model

At chunk 1000 / overlap 0 the models are within one or two cases of each other:

| Model | Q&A Hit@3 | Q&A Hit@5 | Q&A MRR | Style Hit@5 (`llm` query) | Size |
|---|---|---|---|---|---|
| nomic-embed-text | 0.88 | 0.92 | 0.69 | 0.30 | 274 MB |
| nomic-embed-text + prefixes | 0.77 | 0.92 | 0.73 | 0.40 | 274 MB |
| embeddinggemma + prefixes | 0.85 | 0.92 | 0.74 | 0.40 | 621 MB |
| embeddinggemma-2:270m + prefixes | 0.88 | **1.00** | 0.74 | 0.35 | 303 MB |
| qwen3-embedding:0.6b + prefixes | 0.69 | 0.81 | 0.65 | **0.60** | 639 MB |
| mxbai-embed-large + prefixes | 0.81 | 0.85 | 0.71 | 0.50 | 669 MB |

- Q&A: nomic, EmbeddingGemma and EmbeddingGemma 2 are level; Qwen3-Embedding is weakest.
- Style guides: Qwen3-Embedding is best with an LLM query and prefixes, but the gap is a few cases.
- Prefixes: mixed. They help Qwen3-Embedding and EmbeddingGemma 2, not clearly nomic.
- Query time is 10–15 ms for all; indexing all collections took 50–100 s on an RTX 2060 Super.

Switching models means re-indexing every collection, including users' uploads. With these
margins that is not justified yet; fixing the style query and the overlap matters more.

### Finding 4: the guidelines barely change the output

Task model qwen3.5:4b (thinking off, see below), 20 drafts, 3 repeats, scores 0–2.

The judge (`ministral-3:8b`) gave every feedback output 2/2 and sometimes quoted sentences that
were not in the feedback, so feedback was graded by hand instead (one repeat per case, 13 cases).
For grammar the judge agreed with a hand check except st-p-09, where it accepted US-style quotes.

| Variant | Grammar (judge) | Feedback (hand) | Feedback (judge) |
|---|---|---|---|
| fixed guidelines | 1.57 | **1.46** | 2.00 |
| retrieval (as before) | 1.71 | 1.31 | 2.00 |
| no guidelines | 1.57 | **1.46** | 2.00 |

- The model already knows most of what the guidelines say. Fixed guidelines did not beat
  no guidelines, but they also never made the output worse.
- Retrieval sometimes hurts: for the thesis draft (st-f-05) it pulled in a passage that
  encourages first person, and the feedback then recommended it for a formal essay.
- Phrase-level problems are found in every variant (intensifiers, contractions, passive voice,
  telling, first-to-third person switch). Problems about structure are often missed: topic
  sentence, transitions, vague email subject line, plot summary instead of review, sudden
  omniscience. The feedback format (quote → problem → fix per phrase) pushes the model toward
  single phrases.
- Grammar fixes comma splices, fused sentences and agreement, but never dangling modifiers
  (st-p-10) or UK nested quotes (st-p-09), even with the rule in the prompt. Fixing a dangling
  modifier needs rewording, which the grammar prompt forbids ("do not change word choice").
  st-f-02 (redundant words) is not a grammar problem for the same reason and should move to
  feedback.

**Bug found on the way:** with thinking on, qwen3.5:4b used up to 3,500 tokens thinking about three
sentences, filled `num_ctx` (4096) and returned an empty answer (59 of 63 grammar and 91 of 117
feedback outputs in the first run). Grammar and feedback now run with thinking off
(`TaskConfig.think=False`); answers take 1–5 s instead of 50–70 s.

### Finding 5: prompt changes help most, but they are not retrieval

Run 3 tried two prompt changes after Finding 4: a fourth feedback heading "Structure" and allowing
grammar to reword. The Structure heading raised hand-graded feedback from 1.46 to 1.85 (one repeat),
the largest gain of the whole evaluation; rewording changed nothing. Both are about the model call,
not retrieval, so **both were removed again** and handed over (see Handover below). Run 3 is kept in
`evaluation/reports/RAG/phase5_run3_new_prompts/`. All later runs use main's prompts.

### Finding 6: the style-guide search works once the query names the right problem

Runs 4 to 8 (November 2026), all with main's prompts, qwen3.5:4b, thinking off. Feedback graded by
hand (0 = misses the known problem, 1 = touches on it, 2 = names it with a usable fix), 3 repeats.
One run of the same setup varies by about ±0.15 (fixed file only: 1.23 in run 5, 1.07 in run 8, same
10 drafts), so smaller gaps are noise.

**Run 4: hybrid with the keyword query is the weakest variant.** Feedback, 13 drafts: fixed file
1.38, none 1.36, retrieval 1.18, hybrid 1.18. The chunks were mostly off-topic and their text leaked
into the feedback (the model reviewed guide sentences as if they were the user's: 12 of 39 hybrid
answers). Example sentences in the fixed files leaked the same way and were removed.

**Run 5: a free-text LLM query helps the search, not the feedback.** The model first writes a
≤12-word search query naming the draft's problem (one extra call, ~1 s). Hit@2 for feedback rose
from 0.00 (keywords) to 0.29, hybrid feedback from 1.18 to 1.31, leakage fell to 5/39. Still below
the fixed file alone (1.41). *The first attempt of run 5 had an eval bug* (the judging pass
regenerated answers with the wrong variant); those files are kept as `INVALID_*` with a README.

**Retrieval options (`try_retrieval_options.py`, no generation, 20 drafts):** keyword search (BM25),
vector + BM25, cross-encoder re-ranking and chunk sizes 500/1000/1500 were compared with three
queries. Feedback Hit@2 at chunk 1000/400:

| Query | Vector | BM25 | Vector + BM25 | Vector + re-rank |
|---|---|---|---|---|
| keywords only | 0.00 | 0.00 | 0.07 | 0.00 |
| query written by the model (run 5) | 0.29 | 0.29 | 0.29 | 0.21 |
| test set answer (best case) | 0.50 | 0.71 | 0.86 | **0.93** |

No search method rescues a poor query; with a good one, every method works. The free-text queries
named the wrong problem ("How to vary sentence structure" for a literature review that only
summarises sources). Results: `retrieval_options.txt` / `.json`.

**Run 6: even the right passages did not improve qwen3.5:4b's feedback.** Fixed file + the right
passages (label query, re-ranked, right passage in the top 2 for 93% of drafts): 1.33, the same as
hybrid (1.31) and fixed (1.41). **Run 7, ministral-3:8b on 6 drafts:** fixed 0.83, right passages
0.94, hybrid 0.89; worse than qwen3.5:4b overall, and no clear gain from the passages either.

**Run 8: label query + re-ranking + marked references (the app now).**
- *Query:* the model picks **one** label from a list of guide topics per task and style
  (`problem_labels` in `src/tasks/feedback.py` / `grammar.py`). The label is the search query.
  Picking from a list is a multiple-choice question, much easier for a small model than writing a
  query. Picking two labels was tried first: the second was usually wrong and filled the slots.
- *Search:* top 30 by vector search, reordered by a cross-encoder
  (`cross-encoder/ms-marco-MiniLM-L-6-v2`, `RERANK_MODEL`). This is an extra local model, not an
  LLM: 22.7M parameters, 88 MB, it only scores how well each passage fits the label (~0.5 s on the
  CPU for 30 passages). Downloaded once from Hugging Face like the guardrail model, then it runs
  offline; the user's text never leaves the machine. Without the file the app keeps the vector order.
  It raises feedback hits from 4/14 to 9/14 (notebook 8, §4.4).
- *Prompt:* each passage is marked `REFERENCE n, from "<guide>" (advice from a style guide, not the
  user's text; do not review or quote it)`.

Results on 10 feedback drafts (the 4 drafts every variant always got right were left out):

| | Right label picked | Right passage in top 2 | Feedback (hand) | Leakage |
|---|---|---|---|---|
| Free-text query, vector (run 5) | ~2/10 | ~3/10 | 1.10 | 4/30 |
| **Label query, re-ranked, references (run 8)** | **7/10** | **7/10** | **1.13** | 3/30 |
| Fixed file only (run 8 / run 5) | – | – | 1.07 / 1.23 | 2/30 / 1/30 |

Grammar, 6 drafts (judge, st-p-09 corrected by hand to 0): hybrid 1.33, fixed 1.33.

- The style-guide search now finds the right passage for 7 of 10 drafts, and the feedback with it is
  level with the fixed file instead of below it.
- It clearly helps the literature review (6/6 vs 2/6), thesis and first/second person drafts.
- It still hurts two drafts: the email subject line (0/6 vs 4/6) and the book review, where the
  *Book Reviews* guide talks to its reader ("You can and should…") and the model reviewed those
  sentences as the user's text in all 3 answers, despite the REFERENCE label.
- With qwen3.5:4b the guides add little to what the model already knows (Finding 4). Whether a
  larger model makes more of them is the next test.

Evidence: `phase5_run4_*` to `phase5_run8_*`, `label_queries_check.txt`, `retrieval_options.*`.

### Decisions (November 2026)

| Task | Prompt gets | Search query |
|---|---|---|
| Q&A | 3 passages from `userdocs` + 1 from `outline` | the user's question |
| Feedback | fixed guideline file + 2 style-guide passages (`formal`/`fiction`) + 1 `outline` passage | style guide: 1 label picked by the model; outline: the user's text |
| Grammar | fixed guideline file + 3 style-guide passages | 1 label picked by the model |

- **Q&A:** user documents and outlines are split with no overlap (`UPLOAD_CHUNK_OVERLAP=0`). Style
  guides keep `CHUNK_OVERLAP=400`. Documents uploaded before the change keep their old chunks until
  they are uploaded again.
- **Grammar and feedback (hybrid):** the fixed guideline file (`src/tasks/guidelines/{task}_{style}.md`,
  ~200–300 words of core rules taken from the guides) makes sure the basics are always there; the
  style-guide search adds the passage for the draft's specific problem (Finding 6). Style-guide
  passages are re-ranked and marked as references. The search is on by default and can be switched
  off per task with `collection=None` in `src/config.py`.
- **Thinking off** for grammar and feedback (empty answers otherwise, see Finding 4).
- **Context window** configurable with `NUM_CTX` (default 4096; `.env` for development: 8192). The
  eval prompts never used more than ~1,700 tokens, so this did not change any result.
- **Embedding model:** stays `nomic-embed-text`.

### Still open

- **Larger task model** (colleague's test): does it use the style-guide passages better? Run
  `run_task_eval.py --model <model> --variants hybrid,fixed --tasks feedback --no-judge` and grade by
  hand; `--variants oracle` shows the upper bound with the right passages.
- **Leakage from guides that address the reader** (*Book Reviews*). Options: drop such passages, or
  strip second-person sentences from references.
- The label lists were written from the guides' topics, and the test set's problems come from the
  same guides, so the label query may look better on this set than on real drafts. A few labels have
  no test case (jargon, weak conclusion, dialogue attribution, …).
- The `fiction` collection also holds six genre style excerpts (`data/styles/fiction/*.md`, about 150
  words each, 6 of 3,073 chunks). Checked (`check_fiction_excerpts.py`): small effect, left as is.
- Grammar never fixes dangling modifiers or UK nested quotes (Finding 5, Handover).
- The LLM judge is not reliable for feedback. Any later feedback eval needs hand grading or a
  stronger judge.

## Handover: observations about the model call (not changed here)

This evaluation is about retrieval: does the right text reach the model? While checking the final
outputs we saw things that belong to the model call and the prompts. We tried two prompt changes
in run 3 and **removed them again**, because prompts are tested separately. The runs are kept as
evidence in `evaluation/reports/RAG/phase5_run3_new_prompts/`.

**1. Feedback misses problems of the text as a whole.** The output format asks for
`"exact phrase" → issue → fix` under Tone, Style and Clarity. Problems in single phrases are found
reliably (intensifiers, contractions, passive voice, telling). Problems that aren't in one phrase
are often missed: missing thesis (st-f-05), missing topic sentence (st-f-06), missing transitions
(st-f-07), a review that only retells the plot (st-f-10), a narrator who suddenly knows everyone's
thoughts (st-p-07). *Tried in run 3:* a fourth heading "Structure" (thesis, topic sentences,
transitions / point of view, scene versus summary). Hand-graded feedback score rose from 1.46 to 1.85
(13 drafts, one repeat), and most of those problems were found. Removed again; for your evaluation.

**2. Grammar never fixes dangling modifiers.** "Walking through the forest at dusk, the trees seemed
to lean closer" (st-p-10) needs a subject ("…, Mara felt the trees lean closer"), but the prompt says
"do not change meaning, word choice or names". *Tried in run 3:* allow rewording where a sentence
can't be fixed otherwise. No effect: the model still returned the sentence unchanged in all 12 tries.
Removed again. A larger grammar model is the next thing to try. UK nested quotes (st-p-09) are also
never converted, even with the rule in the guidelines.

**3. Thinking was turned off for grammar and feedback (kept, `TaskConfig.think=False`).** qwen3.5:4b
writes its reasoning as tokens before the answer. Prompt, reasoning and answer share the context
window, which the app set to 4,096 tokens (`num_ctx`, now `NUM_CTX` in `.env`) to keep VRAM low. For
three sentences the model thought for ~3,500 tokens, filled the window and stopped before answering:
150 of 180 answers were empty (run 1), in the app too. With thinking off: 0 empty, ~5 s instead of
~55 s per answer. Turning it back on would need a larger `num_ctx` (more VRAM) or a thinking limit.
Thinking is more likely to pay off for tasks with several reasoning steps (Q&A across chunks,
checking a draft against the outline) than for grammar or feedback.

**4. The judge model can't grade feedback.** `ministral-3:8b` gave almost every feedback answer 2/2,
including empty ones, and quoted sentences that weren't in the feedback. Feedback was graded by hand.
For grammar it agreed with a hand check except st-p-09 (it accepted US-style quotes).
