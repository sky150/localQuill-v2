# localQuill evaluation (`src/evaluation`)

This package tests local models on three app tasks, one `--prompt` each:

- `grammar`: grammar correction. ERRANT compares each output with a hand-written and a LanguageTool reference; embeddings and a G-Eval judge add content and quality scores.
- `qna`: questions about the user's documents, judged against a hand-checked answer sheet. See [Q&A](#qa).
- `feedback`: style and clarity feedback, rerun with last year's DISPRO1 setup and metrics. See [Feedback](#feedback).

`config.py` holds each prompt's models and settings (`MODELS_BY_PROMPT`, `SETTINGS_BY_PROMPT`). The settings vary temperature with `top_p` fixed at 1.0. gpt-5-nano takes no temperature and varies the reasoning effort instead. Each setting runs `N_RUNS` (3) times. Temperature 0 runs once, since greedy decoding gives the same output every time.

Run every module from the repo root as `uv run python -m src.evaluation.<module>`.

## Run order

Every script needs `--style formal|fiction` and `--prompt grammar|qna|feedback`. Feedback runs on `formal` only.

| Prompt | Steps (`config.STEPS_BY_PROMPT`) |
|---|---|
| `grammar` | `build_references`, `run_models`, `eval_errant`, `eval_embedding`, `eval_judge`, `aggregate` |
| `qna` | `build_qna_index`, `run_models`, `eval_judge`, `aggregate` |
| `feedback` | `run_models`, `eval_judge`, `aggregate`, then `compare_dispro1` by hand |

0. **Prepare the texts by hand**, per style, in `data/eval/<style>/`:
   - `error_texts/<id>.txt`: the text with errors, under 2000 words.
   - `n_errors.json`: injected error count per text, `{"<id>": 36}` (optional).
   - `opus/<id>.txt`: the corrected text alone. en-GB; fix grammar, punctuation, tense and structure; keep meaning, word choice, names and line layout.
1. **`build_references`** writes LanguageTool references to `languagetool/` and lists missing opus files. It skips existing files, so delete them to rebuild. The local mode (default) needs Java 17+, downloads about 250 MB on first use and takes 1-2 GB of RAM. `--lt-public` uses the public API instead: it waits 10 s per text and applied at most 30 corrections per text in our tests.
2. **`run_models`** generates the outputs into `runs.jsonl`. `--texts <id> ...` limits the run to those texts.
3. **`eval_errant`, `eval_embedding`, `eval_judge`** score the outputs, in any order.
4. **`aggregate`** writes `summary.csv` and `summary_long.csv`.
5. **`log_mlflow`** logs one MLflow run per model and setting to `MLFLOW_TRACKING_URI` (default `http://127.0.0.1:5001`). Start the server with `docker compose up -d mlflow`. Run it last: it exits if a results file is missing.
6. **`ranking_rag_norag`** (optional) plots every metric for RAG and no RAG side by side and writes a composite ranking per condition. It always reads both conditions and takes no `--norag`. See [Composite ranking](#composite-ranking).

```sh
uv run python -m src.evaluation.build_references  --style formal --prompt grammar
uv run python -m src.evaluation.run_models        --style formal --prompt grammar
uv run python -m src.evaluation.eval_errant       --style formal --prompt grammar
uv run python -m src.evaluation.eval_embedding    --style formal --prompt grammar
uv run python -m src.evaluation.eval_judge        --style formal --prompt grammar
uv run python -m src.evaluation.aggregate         --style formal --prompt grammar
uv run python -m src.evaluation.log_mlflow        --style formal --prompt grammar
uv run python -m src.evaluation.ranking_rag_norag --style formal --prompt grammar   # after aggregate --norag too
```

`run_all` runs a prompt's steps for each of its styles, or for the styles you pass, and stops at the first failure. `--skip` drops steps. You run `log_mlflow` by hand.

```sh
uv run python -m src.evaluation.run_all                                  # grammar, formal and fiction
uv run python -m src.evaluation.run_all --style formal --skip build_references
uv run python -m src.evaluation.run_all --norag --skip build_references  # no-RAG baseline
uv run python -m src.evaluation.run_all --prompt qna                     # formal and fiction
uv run python -m src.evaluation.run_all --prompt feedback                # formal only
uv run python -m src.evaluation.run_all --prompt qna --provider openai   # gpt-5-nano
```

- **Results** go to `data/eval/results/<prompt>/<style>/`. Each step resumes where it stopped.
- **`--norag`** runs the no-RAG baseline: `WRITING GUIDELINES` stays empty and results go to `<style>-norag/`. Pass it to every step, `log_mlflow` included; `run_all --norag` does that for you.
- **`--provider`** reaches `run_models` only. Later steps read the rows of every provider.
- **Dependencies:** errant, language_tool_python, mlflow and the pinned `en_core_web_sm` wheel sit in the dev group. `uv sync` installs them, no `spacy download` needed.

## Generation

`run_models` builds the app's task with `src.llm.factory.get_model` and calls `task.model.invoke(prompt)`. It skips `Task.run()`, which adds a random canary, cuts the input at 300 words and drops the token counts.

- **Provider:** `config.PROVIDER` or `--provider` picks `ollama`, `lmstudio` or `openai`. The eval ignores `LLM_PROVIDER` in `.env`.
- **Context:** `LLM_CONTEXT` sets `NUM_CTX` (our runs: 8192). LM Studio fixes the context at load time, so `ensure_context` reloads the model with `NUM_CTX` and raises `ContextMismatch` if that fails. Turn LM Studio's idle TTL off during a run.
- **Sampling:** local requests send `temperature`, `top_p`, seed `SEED_BASE + run` and the output cap `NUM_PREDICT` (4096; the first runs used 2048). OpenAI requests send `reasoning_effort` and a 32768 cap.
- **Warm-up:** one short call loads the model before run 0, so load time stays out of the timings. OpenAI skips it.
- **Prompt:** the app's task template as one user message, without the guardrail canary. `rag_cache.json` freezes the retrieved guides so every model gets the same prompt; delete the file if it ends up empty. `clean_output` strips thinking blocks, a wrapping code fence, the "UNKNOWN WORDS:" trailer and wrapping triple quotes.

## Data layout

`common.EvalPaths(style, prompt, rag)` resolves every path.

```
data/eval/
  <style>/                         formal | fiction (also the Chroma collection name)
    raw/                           source files; feedback reads raw/*.txt (DISPRO1's texts)
    error_texts/<id>.txt           test texts with errors (grammar, qna)
    n_errors.json                  {"<id>": injected error count}, optional
    opus/<id>.txt                  hand-written reference correction
    languagetool/<id>.txt          LanguageTool reference (build_references)
    agreement.csv                  LT vs. opus edit overlap per text (eval_errant)
    qna/<id>/                      Q&A material of one text
      userdocs/*.md                up to 3 other documents of the user
      outline/outline.md           outline or assignment criterion
      answers.json                 {"<question_id>": ["fact", ...]}, 2-4 facts each
  chroma_qna/                      eval-only Chroma for Q&A (build_qna_index, gitignored)
  prompts/<prompt>/
    rag_cache.json                 frozen retrieved chunks
    judge_steps.json               G-Eval steps per criterion (hand-editable)
    questions.json                 qna only
  results/<prompt>/<style>[-norag]/
    runs.jsonl                     one row per generation, input of every scoring step
    errant.csv, embedding.csv, embedding_refs.csv, judge.jsonl, summary.csv, summary_long.csv
    summary_by_question.csv        qna only
    dispro1_comparison.csv         feedback only (compare_dispro1)
    ranking.csv                    composite ranking (ranking_rag_norag)
    rag_vs_norag.png               RAG vs. no-RAG plot, RAG folder only (ranking_rag_norag)
```

A `runs.jsonl` row holds `backend, model, setting_id, params, run, seed, text_id, style, prompt, rag, output, raw_output, wall_s, load_s, prompt_tokens, gen_tokens, gen_tok_per_s, reasoning_tokens, done_reason, num_predict`. Q&A rows add `question_id`, `question` and `history`. `common.run_key` identifies a row by model, setting, run, text and `question_id`.

- `done_reason = "length"`: the output hit the cap and ends mid-text.
- `num_predict`: the row's cap. Older rows lack it, and `config.row_num_predict` reads them as 2048.
- OpenAI rows: `seed` is null, `gen_tokens` includes the hidden reasoning (`reasoning_tokens` gives that part), and `wall_s` includes the network round trip.
- LM Studio and OpenAI rows leave `load_s` and `gen_tok_per_s` empty.

## Modules

| Module | Purpose |
|---|---|
| `config.py` | All settings: styles, prompts, steps, providers, models, settings grids, `N_RUNS`, `SEED_BASE`, `NUM_CTX`, `NUM_PREDICT`, `EXCLUDED_MODELS`, RAG and Q&A limits, judge and embedding models, MLflow |
| `common.py` | Paths (`EvalPaths`), shared I/O, app prompts, RAG retrieval, Ollama calls for judge and embeddings, text normalization |
| `build_references.py` | LanguageTool references; lists missing opus files |
| `build_qna_index.py` | Ingests each text's userdocs and outline into its own Chroma collections; lists missing material |
| `generation.py` | Builds the app's task, sets the LM Studio context, calls the model, keeps token counts and timings |
| `run_models.py` | Model x setting x run x text (x question) loop; resumes via `run_key` |
| `eval_errant.py` | ERRANT edits against the original, per paragraph or for the whole text when paragraph counts differ |
| `eval_embedding.py` | Cosine similarity of original and output (`cos_doc`, `cos_mean`, `cos_min`) |
| `eval_judge.py` | G-Eval judge on Ollama; one `JudgeSpec` per prompt; refuses a judge that is also under test |
| `aggregate.py` | Per-run values, then mean, median and std over runs; drops `EXCLUDED_MODELS` |
| `compare_dispro1.py` | Feedback only: this year's scores per text next to DISPRO1's |
| `log_mlflow.py` | One MLflow run per model x setting; adds new runs on every call |
| `redo_capped.py` | Grammar only: removes rows cut off below the current cap so `run_models` regenerates them |
| `ranking_rag_norag.py` | RAG vs. no-RAG plot and composite ranking for grammar, qna and feedback (`WEIGHTS_BY_PROMPT`, `GATES_BY_PROMPT`) |

## OpenAI (gpt-5-nano)

`--provider openai` runs gpt-5-nano for grammar and Q&A. Feedback has no OpenAI models (`UnsupportedProvider`).

- Set `OPENAI_API_KEY` in `.env`. The model name comes from `config.MODELS_BY_PROMPT`.
- Settings `effort_low` and `effort_medium`, 3 runs each. `ranking.csv` leaves their temperature empty, and the plots draw them as horizontal lines.
- The 32768 cap covers the hidden reasoning tokens.
- OpenAI gets no seed, so runs of one effort differ.
- The test texts, guide chunks and Q&A documents leave your machine.

```sh
uv run python -m src.evaluation.run_all --prompt grammar --provider openai --skip build_references
uv run python -m src.evaluation.run_all --prompt grammar --provider openai --skip build_references --norag
uv run python -m src.evaluation.run_all --prompt qna --provider openai
uv run python -m src.evaluation.run_all --prompt qna --provider openai --norag
```

Then run `log_mlflow` and `ranking_rag_norag` as for the local models.

## Output cap

The first runs capped local output at 2048 tokens to stop looping models. That cap also cut off real output:

- **Fiction grammar:** 181 of 300 RAG rows and 186 of 300 no-RAG rows. The four longest texts need about 1800-2500 output tokens. The cut-off outputs lost about their last tenth, which lowered `tp_rate`, `tp_rate_consensus`, `f05_best` and `cos_doc` most.
- **Formal grammar:** 49 RAG and 22 no-RAG rows, mostly loops (granite4.2:8b).
- **Q&A:** qwen3.5:4b hit the cap in 775 of 960 generations and left 700 answers empty. The QA task leaves `think` unset, and qwen3.5 thinks by default. No other Q&A model hit the cap.

What we did:

- **Grammar:** rerun of the cut-off rows with `NUM_PREDICT = 4096`. The longest prompt (4035 tokens) plus 4096 fits the 8192 context. Ollama reproduces a generation from the same seed, model, context and prompt, so the rows that finished under 2048 would come out the same, and only the cut-off rows need a rerun. This holds while Ollama, the model digests and the context stay unchanged; `--report` checks it.
- **Q&A:** no rerun. qwen3.5:4b left `MODELS_BY_PROMPT["qna"]`, and `EXCLUDED_MODELS` keeps its old rows out of `aggregate` and `log_mlflow`.

`redo_capped` archives the stale rows to `capped_rows.jsonl`, backs up `runs.jsonl` and `judge.jsonl` once (`*.before_redo.jsonl`), and rewrites both files without those rows. It refuses Q&A and feedback. Run it per style, and again with `--norag`:

```sh
uv run python -m src.evaluation.redo_capped --style fiction --prompt grammar --dry-run   # list the rows, change nothing

# Fiction
uv run python -m src.evaluation.redo_capped --style fiction --prompt grammar             # archive and remove them
uv run python -m src.evaluation.run_all --style fiction --prompt grammar --skip build_references
uv run python -m src.evaluation.redo_capped --style fiction --prompt grammar --report    # check the rerun

# Fiction norag
uv run python -m src.evaluation.redo_capped --style fiction --prompt grammar --norag
uv run python -m src.evaluation.run_all --style fiction --prompt grammar --skip build_references --norag
uv run python -m src.evaluation.redo_capped --style fiction --prompt grammar --report --norag

# Formal
uv run python -m src.evaluation.redo_capped --style formal --prompt grammar
uv run python -m src.evaluation.run_all --style formal --prompt grammar --skip build_references
uv run python -m src.evaluation.redo_capped --style formal --prompt grammar --report

# Formal norag
uv run python -m src.evaluation.redo_capped --style formal --prompt grammar --norag
uv run python -m src.evaluation.run_all --style formal --prompt grammar --skip build_references --norag
uv run python -m src.evaluation.redo_capped --style formal --prompt grammar --report --norag
```

`--report` prints per model how many rows came back, how many start with the archived output, and how many hit 4096 again. A prefix mismatch at temperature 0 means the setup changed.

`log_mlflow` only adds runs, so the old 2048 runs stay in MLflow. Filter on the param `num_predict` or delete them by hand.

Check `capped_rate` in `summary.csv` before you compare local models with gpt-5-nano. Values near 0 make the comparison fair.

## Q&A

Four models up to 4B (gemma4:e2b, gemma4:e4b, phi4-mini:3.8b, gemma3:4b) answer questions about each test text at temperatures 0, 0.2, 0.7 and 1.0; gpt-5-nano runs at two reasoning efforts. `data/eval/prompts/qna/questions.json` holds the questions:

| Style | Questions (in order) |
|---|---|
| formal | `summary` "Summarise the text.", `arguments` "What arguments are used?", `arguments_main` "What is the most important argument used?" (follow-up) |
| fiction | `characters` "What characters are in the scene?", `characters_main` "Which of them is the most important?" (follow-up), `summary` "Summarise the text." |

- **Material:** write `data/eval/<style>/qna/<id>/` by hand: up to 3 userdocs, one outline, and `answers.json`. Take the facts from the first 4000 characters of the text, since the model sees no more (`QNA_DOC_CHARS`). After you edit a document, delete `data/eval/chroma_qna/` and `prompts/qna/rag_cache.json`, then rebuild.
- **Prompt:** the app's `QATask` template. `USER DOCUMENTS` and `USER OUTLINE` get the top 3 and top 1 chunks for the question. A follow-up carries the model's own earlier answer. `--norag` sets userdocs and outline to "none found".
- **Judge:** Nemotron scores `correctness` (states the answer-sheet facts, contradicts none) and `groundedness` (claims nothing the documents don't say), 1 to 5.
- **Output:** `summary.csv` averages over texts and questions; `summary_by_question.csv` splits by question.

Run `build_qna_index` first and fix what it lists. Start the MLflow server before `log_mlflow`.

```sh
uv run python -m src.evaluation.build_qna_index --style formal --prompt qna   # check the material
uv run python -m src.evaluation.build_qna_index --style fiction --prompt qna
uv run python -m src.evaluation.run_all --prompt qna                          # formal and fiction
uv run python -m src.evaluation.run_all --prompt qna --norag                  # no-RAG baseline
uv run python -m src.evaluation.log_mlflow --style formal --prompt qna
uv run python -m src.evaluation.log_mlflow --style fiction --prompt qna
uv run python -m src.evaluation.log_mlflow --style formal --prompt qna --norag
uv run python -m src.evaluation.log_mlflow --style fiction --prompt qna --norag
uv run python -m src.evaluation.ranking_rag_norag --style formal --prompt qna   # after aggregate --norag too
uv run python -m src.evaluation.ranking_rag_norag --style fiction --prompt qna
```

## Feedback

Feedback reruns `ministral-3:8b`, last semester's best feedback model, with DISPRO1's setup: the 9 original formal texts in `data/eval/formal/raw/*.txt`, temperature 0.1, top-k 3 guide chunks and `llama3.1:8b` as judge. DISPRO1's fiction texts no longer exist, so fiction stays out.

- **Prompt:** the app's `FeedbackTask` template with frozen guides. The label query and re-ranker stay out; the outline reads "none found".
- **Judge:** DISPRO1's G-Eval metrics `style` and `clarity`, 0 to 10, stored as 0-1 in `judge.jsonl`. DISPRO1's grammar metric stays out, since the feedback task no longer covers grammar.
- **Comparison:** `compare_dispro1` writes `dispro1_comparison.csv` with style, clarity, time and tokens per text, now and then. DISPRO1's `duration_seconds` may include the judge calls; check `data/eval/formal/raw/DISPRO1_Report.pdf` before you compare times.

```sh
uv run python -m src.evaluation.run_all --prompt feedback
uv run python -m src.evaluation.compare_dispro1 --style formal --prompt feedback
uv run python -m src.evaluation.ranking_rag_norag --style formal --prompt feedback   # after aggregate --norag too
```

## Grammar judge

Nemotron scores two criteria, 1 to 5:

- **correction:** spelling, punctuation, agreement and tense are right; awkward sentences read clearly.
- **preservation:** content, fine sentences, voice, names and formatting stay as they were.

Correction rewards edits and preservation penalizes them, so under- and over-editing models show different profiles. Edit the rubric in `judge_steps.json`, or delete a key to let the judge write new steps. With logprobs, the score is the probability-weighted mean over 1-5 (`logprob_weighted`); without, the judge samples `JUDGE_SAMPLES` times and keeps the median.

## Metrics

### Shorthand

| Term | Meaning |
|---|---|
| hyp | the tested model's corrected text |
| ref, LT, opus | reference correction: LanguageTool or the hand-written opus file |
| edit | one ERRANT change: token span `(o_start, o_end)` plus replacement. Edits match when both parts match |
| span | the original token range an edit touches |
| TP, FP, FN | model edit a reference made; model edit no reference made; reference edit the model missed |
| F0.5 | `1.25 * P * R / (0.25 * P + R)`, weighs precision twice as much as recall |
| best | per paragraph, the reference that gives the model the higher F0.5 |
| union, cons | edits of at least one reference; edits of both references |

### Aggregation

Each model, setting and run gets one value per metric. ERRANT counts add up over all texts before they become rates, so long texts weigh more. Embedding, judge and timing values average over texts. `summary_long.csv` gives `mean`, `median`, `std` and `n_runs`; `summary.csv` holds the means.

### summary.csv columns

| Column | Range | Better | Meaning |
|---|---|---|---|
| `tp_rate` | 0-1 | higher | **Primary.** Share of error spans the model fixed like a reference |
| `tp_rate_consensus` | 0-1 | higher | Share of the edits both references agree on that the model made |
| `tp_rate_known` | 0-1+ | higher | `tp_spans / n_errors`. Exceeds 1 when ERRANT splits one error into several edits |
| `precision_best` | 0-1 | higher | Share of model edits the best reference confirms |
| `f05_best` | 0-1 | higher | F0.5 against the best reference |
| `overcorr_per_100tok` | 0+ | lower | Model edits no reference made, per 100 original tokens |
| `cos_doc` | 0-1 | higher | Cosine of mean paragraph embeddings, original vs. output. Measures kept content. A preface like "Certainly! Here is..." lowers it too |
| `cos_mean`, `cos_min` | 0-1 | higher | Mean and lowest paragraph-pair cosine. Empty when paragraph counts differ |
| `judge_correction`, `judge_preservation` | 1-5 | higher | Grammar judge scores |
| `judge_correctness`, `judge_groundedness` | 1-5 | higher | Q&A judge scores |
| `judge_style`, `judge_clarity` | 0-1 | higher | Feedback judge scores |
| `wall_s`, `wall_s_median` | s | lower | Seconds per generation, model loaded |
| `gen_tok_per_s` | tok/s | higher | Output tokens per second of generation. Ollama only |
| `prompt_tokens`, `gen_tokens`, `reasoning_tokens` | tokens | | Mean tokens per generation; `reasoning_tokens` is OpenAI only |
| `capped_rate` | 0-1 | lower | Share of generations that hit the output cap |

`errant.csv` holds the per-generation counts behind these rates (`tp_*`, `fp_*`, `fn_*`, `n_*`, `mode`, `tp_types`). `agreement.csv` compares the two references; a low `n_cons` means `tp_rate_consensus` rests on few edits.

### Composite ranking

`ranking_rag_norag` combines metrics into one `score` per model x setting. `WEIGHTS_BY_PROMPT` and `GATES_BY_PROMPT` at the top of `ranking_rag_norag.py` set the weights and gates per prompt.

1. **Gate:** grammar needs `cos_doc` >= 0.95; Q&A needs `capped_rate` <= its threshold; feedback has no gate. A lower-is-better gate metric passes at or below its threshold, any other at or above. A failing row stays in `ranking.csv` with `passed_gate = False` and no score.
2. **Scale:** min-max to 0-1 over the passing rows, best row 1. Lower-is-better metrics flip.
3. **Weight:** the weighted mean of the scaled metrics a row has. A row without `gen_tok_per_s` (LM Studio, OpenAI) still scores 0-1.

The script warns when fewer than 2 rows pass, as for feedback today (one model, one setting). All-zero weights give NaN scores.

Scores compare only within one `ranking.csv`. Each folder (`<style>` or `<style>-norag`) scales on its own, and adding a model moves every score. Compare RAG with no RAG, or formal with fiction, on the raw metric values: the plot, the printed tables or the metric columns of `ranking.csv`.

### MLflow

Each prompt has its own experiment: "Grammar Evaluation", "Q&A Evaluation", "Feedback Evaluation". `log_mlflow` adds one run per model x setting, named `<prompt>_<style>_<model>_<setting_id>`.

| Kind | Name | Meaning |
|---|---|---|
| tag | `testset_sha`, `backend`, `style`, `prompt` | test set hash (compare runs with equal values), provider, CLI arguments |
| param | `provider`, `model`, `temperature`, `top_p`, `reasoning_effort` | the setting; OpenAI logs `None` for temperature and `top_p` |
| param | `num_predict`, `num_predict_min` | highest and lowest cap among the run's rows |
| param | `style`, `collection`, `top_k`, `n_cases`, `n_runs`, `num_ctx`, `seed_base` | run setup |
| param | `judge_model`, `embed_model`, `references` | scoring setup |
| artifact | `prompt_template.txt`, `results.json` | the task template; one row per generation with input, prompt, output and all scores |
| step metric | `latency_s`, `similarity`, `per_run/<metric>` | per generation or per run |
| summary | `<metric>`, `<metric>_std`, `error_rate`, `latency_mean_s`, `latency_max_s`, `response_chars_mean`, `similarity_mean` | means over runs; the last five match `evaluation/notebooks/8_task_grammer_evaluation.ipynb` |

## Decisions

- **Judge:** NVIDIA Nemotron Nano 9B v2 (`hf.co/GGUF-A-Lot/NVIDIA-Nemotron-Nano-9B-v2-GGUF:Q4_K_M`), from no family under test. Grammar uses two criteria instead of four, which halves the judge calls. Feedback keeps DISPRO1's judge `llama3.1:8b`, so its scores compare with last year's.
- **Settings:** grammar 0 to 0.3, Q&A 0, 0.2, 0.7 and 1.0, feedback 0.1; `top_p` 1.0.
- **Cloud baseline:** gpt-5-nano on grammar and Q&A, efforts low and medium, cap 32768.
- **Limits:** output cap 4096 and context 8192 for every local model.
- **Grammar models:** granite4.2:8b, qwen3.5:9b, llama3.1:8b, mistral:7b-instruct, gemma4:e4b.
- **Q&A models:** gemma4:e4b, gemma4:e2b, phi4-mini:3.8b, gemma3:4b; qwen3.5:4b excluded.
- **Pipeline tests:** qwen2.5:0.5b (Ollama) and qwen2.5-coder-0.5b-instruct (LM Studio).
- **Test data:** 10 formal and 7 fiction texts under 2000 words, errors injected with Claude.
