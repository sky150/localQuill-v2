# localQuill grammar evaluation (`src/evaluation`)

This package tests local grammar-correction models against hand-written and LanguageTool references. The settings grid varies temperature alone: 0, 0.1, 0.2 and 0.3, with `top_p` fixed at 1.0. Each setting runs `N_RUNS` (5) times. Temperature 0 picks the most likely token at each step and gives the same output every time, so it runs once (`config.runs_for`). The report gives mean, median and std over runs.

Run every module from the repo root as `uv run python -m src.evaluation.<module>`. The modules hold functions, plus a few frozen dataclasses for typed returns.

## Run order

Every script takes two required arguments that pick the folders: `--style formal|fiction` and `--prompt grammar|feedback|qna`. Only `grammar` runs today. `feedback` and `qna` have folders, but the scripts exit with `UnsupportedPrompt` until those prompts get a prompt builder and metrics.

0. **Prepare the texts by hand**, per style:
   - the text with errors in `data/eval/<style>/error_texts/<id>.txt`, under 2000 words;
   - its injected error count in `data/eval/<style>/n_errors.json` (`{"<id>": 36}`, optional);
   - the corrected version in `data/eval/<style>/opus/<id>.txt`. It holds the corrected text alone and follows the app's grammar prompt rules: en-GB; change grammar, punctuation, tense and structure; keep meaning, word choice, names and the line layout.

   Source material still to prepare sits in `data/eval/<style>/raw/`.
1. **`build_references`** writes the LanguageTool references to `data/eval/<style>/languagetool/` and lists missing opus files. It never overwrites an existing file, so delete the old files before you rebuild. Two modes:
   - local (default): needs Java 17 or newer. `language_tool_python` downloads the server on first use (about 250 MB) and takes 1-2 GB of RAM while it runs. Use this mode: it has no rate limit and keeps one rule version.
   - `--lt-public`: the public API. It pauses `LT_PUBLIC_DELAY_S` (10 s) after each text to stay under the rate limit, and in our tests it applied at most 30 corrections per text.
2. **`run_models`** generates the corrections and appends them to `runs.jsonl`. See [Generation](#generation).
3. **`eval_errant`, `eval_embedding`, `eval_judge`** score the generations. They don't depend on each other, so run them in any order.
4. **`aggregate`** writes `summary.csv` and `summary_long.csv`.
5. **`log_mlflow`** publishes one MLflow run per model and setting to the experiment "Grammar Evaluation" at `MLFLOW_TRACKING_URI` (default `http://127.0.0.1:5001`). Start the server with `docker compose up -d mlflow`. Run it last: it exits if a results file is missing.
6. **`rag_norag_best`** (optional) reads the `summary_long.csv` of the RAG and the no-RAG run of one style. It plots each metric over temperature with RAG on the left and no RAG on the right, and writes a composite ranking per condition. It takes `--style` and `--prompt` but no `--norag`, since it always reads both conditions. A missing condition is skipped. See [Composite ranking](#composite-ranking).

```sh
uv run python -m src.evaluation.build_references --style formal --prompt grammar
uv run python -m src.evaluation.run_models       --style formal --prompt grammar
uv run python -m src.evaluation.eval_errant      --style formal --prompt grammar
uv run python -m src.evaluation.eval_embedding   --style formal --prompt grammar
uv run python -m src.evaluation.eval_judge       --style formal --prompt grammar
uv run python -m src.evaluation.aggregate        --style formal --prompt grammar
uv run python -m src.evaluation.log_mlflow       --style formal --prompt grammar
uv run python -m src.evaluation.rag_norag_best   --style formal --prompt grammar   # after aggregate --norag too
```

`run_all` runs steps 1 to 4 in order for every style, or for the styles you pass to `--style`. It stops at the first failing step. `--skip` leaves out the steps you name, and `log_mlflow` stays a manual step:

```sh
uv run python -m src.evaluation.run_all                                  # grammar, formal and fiction
uv run python -m src.evaluation.run_all --style formal --skip build_references
uv run python -m src.evaluation.run_all --norag --skip build_references  # no-RAG baseline
```

Steps 1 to 4 write their results to the gitignored folder `data/eval/results/<prompt>/<style>/`, and each step resumes where it stopped. Every (prompt, style) pair gets its own folder, so formal and fiction results stay apart.

`--norag` runs the no-RAG baseline. Every script accepts it. The prompt keeps the task template but leaves `WRITING GUIDELINES` empty, and the results go to `data/eval/results/<prompt>/<style>-norag/`. Pass `--norag` to every step of a baseline run, including `log_mlflow`; `run_all --norag` does this for you. The MLflow run name then carries `<style>-norag`, and the run has a `rag` tag and param. `runs.jsonl` keys rows by model, setting, run and text, not by provider: if you switch provider, the model names differ, so the rows stay apart, but every later step reads both providers' rows.

Dependencies: errant, language_tool_python, mlflow and the `en_core_web_sm` wheel (pinned in `[tool.uv.sources]`) sit in the dev group of `pyproject.toml`. `uv sync` installs them; you need no `spacy download`.

## Generation

`run_models` builds the app's task class (`GrammarTask`) with the model from `src.llm.factory.get_model`, so you test the model the app would build. `generation.py` then calls `task.model.invoke(prompt)` itself. It skips `Task.run()`, which adds a random canary, cuts the input at `GRAMMAR_MAX_WORDS` (300), appends a "Response time" line and drops the token counts.

- **Provider.** `config.PROVIDER` picks `ollama` or `lmstudio`; `--provider` overrides it. The eval ignores `.env`'s `LLM_PROVIDER`, which configures the app. `config.MODELS_BY_PROVIDER` lists the default models of each provider, since Ollama tags and LM Studio ids differ.
- **Context size.** `LLM_CONTEXT` in `.env` (default 16384) sets `NUM_CTX` for both providers. Ollama receives it with every request. LM Studio fixes the context when it loads a model, so `ensure_context` reads the loaded instances through LM Studio's REST API, unloads any instance with another context and loads the model with `NUM_CTX`. It raises `ContextMismatch` if LM Studio still reports another size. Keep the model's idle TTL in LM Studio off during a run: an auto-unloaded model reloads with LM Studio's default context.
- **Sampling.** Each request sends `temperature`, `top_p`, the seed (`SEED_BASE + run`) and the output cap `NUM_PREDICT` (2048 tokens). Small models can loop without stopping; the cap ends them, and their rows show `done_reason = "length"`.
- **Warm-up.** One short call loads the model before run 0, so load time stays out of the timings.

### Prompt

Models get the app's `GrammarTask` template (`src/tasks/grammar.py`) as one user message. The guardrail canary note stays out: its random canary would give every model a different prompt. The RAG query matches the app's, `retrieval_query` plus the first 2000 characters of the text, and `rag_cache.json` freezes the retrieved guides so every model sees the same prompt. If the cache file ends up empty, delete it; the next run rebuilds it from Chroma. `clean_output` strips thinking blocks, a wrapping code fence, the "UNKNOWN WORDS:" trailer and wrapping triple quotes. Importing `src.tasks` adds about a minute of startup time.

## Data layout

`common.EvalPaths(style, prompt, rag)` resolves every path. `EvalPaths.variant` is `<style>`, or `<style>-norag` when `rag` is off.

```
data/eval/
  <style>/                         formal | fiction (also the Chroma collection name)
    raw/                           source files still to prepare (no script reads them)
    error_texts/<id>.txt           test texts with errors; the file name is the text id
    n_errors.json                  {"<id>": injected error count}, optional
    opus/<id>.txt                  hand-written reference correction
    languagetool/<id>.txt          LanguageTool reference (build_references)
    agreement.csv                  LT vs. opus edit overlap per text (eval_errant)
  prompts/<prompt>/                grammar | feedback | qna
    rag_cache.json                 frozen top-k guides per "<style>/<id>|k=<k>" (auto)
    judge_steps.json               frozen G-Eval steps per criterion (hand-editable)
  results/<prompt>/<style>[-norag]/  gitignored; -norag holds the no-RAG baseline
    runs.jsonl                     one row per generation, the input of every scoring step
    errant.csv, embedding.csv, embedding_refs.csv, judge.jsonl, summary.csv, summary_long.csv
    ranking.csv                    composite ranking of this folder's runs (rag_norag_best)
    rag_vs_norag.png               RAG vs. no-RAG plot, written to the RAG folder only (rag_norag_best)
```

A `runs.jsonl` row holds `backend, model, setting_id, params, run, seed, text_id, style, prompt, rag, output, raw_output, wall_s, load_s, prompt_tokens, gen_tokens, gen_tok_per_s, done_reason`.

- `backend` names the provider: `ollama` or `lmstudio`.
- `done_reason` is Ollama's `done_reason` or LM Studio's `finish_reason`. `length` in both means the output hit `NUM_PREDICT` and ends mid-text.
- LM Studio reports neither load time nor decode time, so its rows leave `load_s` and `gen_tok_per_s` empty.

A vLLM runner only has to write the same fields with `backend="vllm"`.

## Modules

| Module | Purpose | Key functions |
|---|---|---|
| `config.py` | All settings: `EVAL_DATA_DIR`, `STYLES`, `PROMPTS`, `SUPPORTED_PROMPTS`, hosts, `PROVIDER`, `MODELS_BY_PROVIDER`, `SETTINGS`, `N_RUNS=5`, `SEED_BASE`, `NUM_CTX` (from `LLM_CONTEXT`), `NUM_PREDICT`, `RAG_TOP_K=3`, `LT_LANGUAGE`, `LT_PUBLIC_DELAY_S`, `EMBED_MODEL`, `EVAL_JUDGE_MODEL` (default the Nemotron hf.co tag), `MLFLOW_*` | `runs_for` |
| `common.py` | Eval paths, shared I/O, app prompt, RAG, Ollama calls for judge and embeddings, text normalization | `EvalPaths`, `eval_paths`, `add_target_args`/`paths_from_args`, `UnsupportedPrompt`, `load_testset`, `testset_sha`, `load_reference`, `load_runs`, `run_key`, `read_jsonl`/`append_jsonl`, `clean_output`, `normalize_for_scoring` (unwraps hard line breaks, collapses spaces, straightens curly quotes), `split_paragraphs`, `ollama_chat`, `ollama_embed`, `TASK_BY_PROMPT`, `retrieve_guides`, `build_messages` |
| `build_references.py` | LanguageTool reference (first suggestion per match, never overwrites a file); lists missing opus files | `build_languagetool`, `report_missing` |
| `generation.py` | Builds the app's task with the factory model, sets the LM Studio context, invokes the model and keeps token counts and timings; per-provider request options (Ollama `options`; LM Studio `max_tokens`, `seed`) | `build_task`, `ensure_context`, `generate`, `warm_up`, `Generation`, `UnknownProvider`, `ContextMismatch` |
| `run_models.py` | Model x setting x run x text loop; resumes via `run_key` | `main` (`--style`, `--prompt`, `--norag`, `--provider`, `--models`, `--runs`) |
| `eval_errant.py` | ERRANT edits against the original text; edit key `(o_start, o_end, c_str)` per paragraph; compares paragraph by paragraph, or the whole text if paragraph counts differ (`mode` column) | `parse`/`edits` (lru-cached), `segment`, `f05`, `score_item`, `reference_agreement` |
| `eval_embedding.py` | Cosine similarity of original and output: `cos_doc` (mean-pooled), `cos_mean`/`cos_min` (paragraph pairs); reference calibration in `embedding_refs.csv` | `embed` (cached, batches of 32), `similarity` |
| `eval_judge.py` | G-Eval judge on Ollama; sees the original and the output, nothing else. Two criteria, two calls per generation. Refuses a judge model listed in `MODELS_BY_PROVIDER` | `load_steps`, `score_prompt`, `weighted_score`, `score` (`--runs`, `--criteria`) |
| `aggregate.py` | Per run, sums ERRANT counts over texts, then computes rates; then mean, median and std over runs | `errant_metrics`, `per_run_tables`, `summarize` (returns tables, writes nothing) |
| `log_mlflow.py` | Reads the results folder and `summarize()`, logs one MLflow run per model x setting; adds new runs on every call | `load_extras`, `log_group`, `main` |
| `rag_norag_best.py` | RAG vs. no-RAG plot per metric and composite score per condition; writes `rag_vs_norag.png` and `ranking.csv`, opens no window | `load_grouped`, `plot`, `metric_ranking`, `composite` (`WEIGHTS`, `GATES`) |

## Judge

The judge scores each generation on two axes:

- **correction** (grammar + structure): spelling, punctuation, agreement and tense are right, and awkward sentences read clearly.
- **preservation** (meaning + style): the content stays the same, fine sentences stay untouched, and voice, names and formatting survive.

Correction rewards changes and preservation penalizes them, so an under-correcting model and an over-editing model get different profiles. `judge_steps.json` holds the rubric steps for each criterion. We merged them by hand from the earlier four criteria; edit them there, or delete a key to let the judge write new steps. When the server returns logprobs, the score is the probability-weighted mean over the digits 1-5 (`method = logprob_weighted` in `judge.jsonl`). Otherwise the judge samples `JUDGE_SAMPLES` times and keeps the median (`sample_median`), which triples the calls.

## Metrics

### Shorthand

| Term | Meaning |
|---|---|
| hyp | hypothesis: the tested model's corrected text |
| ref, LT, opus | reference correction: `languagetool` (LT) or the hand-written `opus` file |
| edit | one ERRANT change from the original: the token span `(o_start, o_end)` and its replacement string. Two edits match when both parts are identical |
| span | the original token range an edit touches, ignoring the replacement |
| TP | true positive: a model edit that a reference also made |
| FP | false positive: a model edit the reference did not make |
| FN | false negative: a reference edit the model did not make |
| P, R | precision `TP / (TP + FP)`, recall `TP / (TP + FN)` |
| F0.5 | `1.25 * P * R / (0.25 * P + R)`: weighs precision twice as much as recall, the usual choice in grammar correction, since a wrong edit hurts more than a missed one |
| best | per paragraph, the reference that gives the model the higher F0.5 (errant_compare style) |
| union | edits made by at least one reference |
| cons | consensus: edits both references made |
| cos | cosine similarity of embedding vectors, 1 = same meaning |
| t / p | `temperature` / `top_p` (nucleus sampling: sample from the smallest token set whose probability sums to p) |

### Aggregation

1. Each (model, setting, run) gets one value per metric. ERRANT counts add up over all texts of the run before they become rates (micro average), so long texts weigh more. Embedding, judge and timing values average over the texts.
2. `summary_long.csv` gives `mean`, `median`, `std` and `n_runs` of those per-run values for each metric. `summary.csv` holds the means, one row per `model x setting_id`.
3. `t0.0` runs once, so its `std` stays empty.

### summary.csv columns

| Column | Range | Better | Meaning |
|---|---|---|---|
| `tp_rate` | 0-1 | higher | **Primary.** `tp_spans / n_err_spans`: share of error spans (spans any reference edited) where the model made the same edit as a reference. Two references fixing one span differently count as one error |
| `tp_rate_consensus` | 0-1 | higher | `tp_cons / n_cons`: share of the edits both references agree on that the model also made. These are the most certain errors |
| `tp_rate_known` | 0-1+ | higher | `tp_spans / n_errors`, against the injected error count. Exceeds 1 when ERRANT splits one error into several edits. Appears when every text has `n_errors` |
| `precision_best` | 0-1 | higher | `tp_best / (tp_best + fp_best)`: share of model edits the best reference confirms |
| `f05_best` | 0-1 | higher | F0.5 from `tp_best`, `fp_best`, `fn_best` |
| `overcorr_per_100tok` | 0+ | lower | `100 * fp_union / orig_tokens`: model edits no reference made, per 100 original tokens. Counts rewrites, added prefaces and made-up text |
| `cos_doc` | 0-1 | higher | cosine of the mean paragraph embedding of original vs. output (`EMBED_MODEL`). Measures kept content, not correctness. The references score 0.99-1.0 on `g_man_vs_computer`; outputs with made-up or missing text drop below 0.8. An empty output scores 0 |
| `cos_mean` | 0-1 | higher | mean cosine of paragraph pairs (paragraph i vs. paragraph i). Empty unless output and original have the same paragraph count |
| `cos_min` | 0-1 | higher | lowest paragraph-pair cosine: catches one rewritten paragraph that the mean hides. Same condition as `cos_mean` |
| `judge_correction` | 1-5 | higher | LLM judge: spelling, punctuation, agreement and tense right; awkward sentences rewritten |
| `judge_preservation` | 1-5 | higher | LLM judge: content unchanged, fine sentences left alone, voice, names and formatting kept |
| `wall_s` | s | lower | mean wall-clock seconds per generation, from request to answer, with the model loaded |
| `wall_s_median` | s | lower | median of the same, robust to one slow call |
| `gen_tok_per_s` | tok/s | higher | output tokens per second of pure generation (`eval_count / eval_duration`). Ollama only |

Known limitation: `cos_doc` averages all output paragraphs, so a preface such as "Certainly! Here is the revised text:" lowers it as if content had changed. Good qwen2.5:0.5b outputs on `g_man_vs_computer` land near 0.88 for this reason.

### Composite ranking

`rag_norag_best` combines several metrics into one `score` per model x temperature:

1. **Gate.** A row with `cos_doc` below 0.95 (`GATES`) has made-up or missing text. It stays in `ranking.csv` with `passed_gate = False` but gets no score and no rank.
2. **Scale.** Each metric in `WEIGHTS` is min-max scaled to 0-1 over the rows that pass the gate: the best row gets 1, the worst 0. Lower-is-better metrics (`overcorr_per_100tok`, `wall_s`) are flipped. A metric with the same value in every row scores 1 for all of them.
3. **Weight.** `score` is the weighted mean of the scaled metrics a row has. It divides by the sum of the weights present, so a row without `gen_tok_per_s` (LM Studio) still scores 0-1, and the weights need not sum to 1.

`ranking.csv` holds one row per model x temperature: `rank`, `model`, `temperature`, `best_for_model` (that model's highest-scoring temperature), `passed_gate`, `score`, then the mean of every summary.csv metric.

**Scores only compare within one folder.** Min-max scaling makes a score relative to the other rows in the same run. Each `ranking.csv` scales over its own folder, `results/<prompt>/<style>/` or `results/<prompt>/<style>-norag/`. So:

- A score of 0.8 for RAG and 0.6 for no RAG does not mean RAG is better. Each side calls its own best row 1 and its own worst row 0.
- Formal and fiction scores don't compare either. The texts, references and error counts differ, and each style is scaled on its own.
- Adding or removing a model changes the min and max, so every other score moves too. Compare ranks within one run, not scores between runs.
- Min-max stretches small gaps. If `f05_best` only spans 0.80 to 0.81, that 0.01 still covers the full 0-1 range.

To compare RAG with no RAG, use the raw metric values instead: the plot, the per-metric tables the script prints, or the metric columns of `ranking.csv`. Both conditions use the same texts and references, so their raw values are comparable. The plot shares the y-axis across each row, so the RAG and no-RAG panels of one metric line up. Raw values from formal and fiction come from different texts, so read a gap between the styles as a hint, not a result.

### Per-generation columns (errant.csv, MLflow `results.json`)

| Column | Meaning |
|---|---|
| `mode` | `paragraph` when output, original and references have the same paragraph count and get compared paragraph by paragraph, else `whole` (one block) |
| `orig_tokens` | spaCy tokens in the original |
| `n_hyp` | edits the model made |
| `tp_any` | model edits that match at least one reference |
| `tp_languagetool`, `tp_opus` | model edits that match that reference |
| `n_languagetool`, `n_opus` | edits that reference made |
| `n_err_spans`, `tp_spans` | error spans, and error spans the model fixed like a reference (see `tp_rate`) |
| `n_cons`, `tp_cons` | consensus edits, and those the model also made |
| `tp_best`, `fp_best`, `fn_best` | TP / FP / FN against the best reference per paragraph |
| `fp_union` | model edits no reference made |
| `n_errors` | injected error count from `n_errors.json` |
| `tp_types` | ERRANT error types of the matched edits, e.g. `{"R:VERB:SVA": 3}` (replace, verb, subject-verb agreement) |

`data/eval/<style>/agreement.csv` compares the two references: `n_lt`, `n_opus`, `n_cons` (edits both made) and `n_errors`. A low `n_cons` next to a high `n_opus` means LanguageTool missed most errors, and `tp_rate_consensus` then rests on few edits.

### MLflow

Experiment "Grammar Evaluation" gets one run per `model x setting_id`, named `<prompt>_<style>_<model>_<setting_id>`. Each call of `log_mlflow` adds new runs.

| Kind | Name | Meaning |
|---|---|---|
| tag | `testset_sha` | first 12 hex chars of a SHA-256 over the names and contents of `error_texts/*.txt`. Compare runs with the same value |
| tag | `backend` | the provider: `ollama` or `lmstudio` |
| tag | `style`, `prompt` | the `--style` and `--prompt` of the run |
| param | `prompt` | the `--prompt`, which names the app task whose template ran |
| param | `provider`, `model` | provider and model name |
| param | `temperature`, `top_p` | the setting |
| param | `style`, `collection` | prompt style and Chroma collection |
| param | `top_k` | guide chunks retrieved per text |
| param | `n_cases`, `n_runs` | texts and runs in this MLflow run |
| param | `num_ctx`, `seed_base` | context window (`LLM_CONTEXT`); run k uses seed `seed_base + k` |
| param | `judge_model`, `embed_model`, `references` | models and references used for scoring |
| artifact | `prompt_template.txt` | the app's `GrammarTask` template |
| artifact | `results.json` | one row per generation: `input` (original), `expected` (opus reference), `prompt` (full prompt sent), `response` (cleaned output), `error`, `latency_s`, `prompt_chars`, `response_chars`, `similarity`, `text_id`, `run`, `seed`, plus that generation's ERRANT, `cos_*` and `judge_*` values |
| step metric | `latency_s` | wall-clock seconds of generation i (step = generation index) |
| step metric | `similarity` | difflib `SequenceMatcher` ratio of generation i against the opus reference, 0-1, 1 = identical. Both sides get whitespace-normalized first, so reflowed lines don't count |
| step metric | `per_run/<metric>` | the per-run value of each summary.csv metric, step = run index. Shows the spread across runs |
| summary | `<metric>`, `<metric>_std` | mean and std over runs of each summary.csv metric |
| summary | `error_rate` | share of generations with an empty output after cleaning |
| summary | `latency_mean_s`, `latency_max_s` | mean and max of `latency_s` |
| summary | `response_chars_mean` | mean output length in characters |
| summary | `similarity_mean` | mean of `similarity` |

`error_rate`, `latency_*`, `response_chars_mean` and `similarity` reuse the names from `evaluation/notebooks/8_task_grammer_evaluation.ipynb`, so you can compare both kinds of runs. The notebook computes `similarity` on raw strings, which gives lower values.

## Decisions so far

- Judge model: NVIDIA Nemotron Nano 9B v2 (`hf.co/GGUF-A-Lot/NVIDIA-Nemotron-Nano-9B-v2-GGUF:Q4_K_M`), about 7 GB. No family under test (Granite, Qwen, Llama, Mistral, Gemma, Phi) serves as judge.
- Judge criteria: two axes, correction and preservation, to halve the judge calls.
- Settings: temperature 0 to 0.3 with `top_p` 1.0. Higher temperatures add nothing for grammar correction.
- Context: 16384 tokens for every model and both providers (`LLM_CONTEXT`).
- Test models: granite4.2:8b, qwen3.5:9b, llama3.1:8b, mistral:7b-instruct (HF version), gemma4:e4b. Q&A (up to 4B): gemma4:e4b, gemma4:e2b, phi4-mini:3.8b, gemma3:4b, qwen3.5:4b. Pipeline tests use qwen2.5:0.5b (Ollama) and qwen2.5-coder-0.5b-instruct (LM Studio).
- Test data: 10 formal and 7 fiction texts (one per fiction genre), each under 2000 words, errors injected with Claude. First item: `data/eval/formal/error_texts/g_man_vs_computer.txt` (30 injected and 6 existing errors, `n_errors=36`).

## Open TODOs

2. Pin `top_k`, `min_p` and `repeat_penalty` in `config.py` for both providers, since Ollama, LM Studio and the model presets use different defaults. Fix the reasoning toggle for models that have one (`google/gemma-4-e4b` defaults to on).
3. Optional: `eval_judge_calibration.py`. Score about 30 outputs with Nemotron and Opus; a Spearman correlation of 0.7 or more means the local judge works.

