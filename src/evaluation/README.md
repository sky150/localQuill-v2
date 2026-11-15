# localQuill grammar evaluation (`src/evaluation`)

Evaluates local grammar-correction models. The settings grid varies temperature only
(0, 0.1, 0.2, 0.3; `top_p` fixed at 1.0). Each setting runs `N_RUNS` (5) times, except
temperature 0, which is deterministic and runs once (`config.runs_for`). The report gives
mean / median / std over runs. All modules are functions (no classes),
run from the repo root as `uv run python -m src.evaluation.<module>`.

## Run order

Every script takes two required arguments that pick the folders: `--style formal|fiction`
and `--prompt grammar|feedback|qna`. Only `grammar` runs today; `feedback` and `qna` have
folders but exit with `UnsupportedPrompt` until they get a prompt builder and metrics.

0. Prepare the texts by hand, per style: put each text with errors in `data/eval/<style>/error_texts/<id>.txt`, its injected error count in `data/eval/<style>/n_errors.json` (`{"<id>": 36}`, optional), and the corrected version in `data/eval/<style>/opus/<id>.txt`. The opus file holds the corrected text only and follows the app's grammar prompt rules: en-GB; change only grammar, punctuation, tense and structure; keep meaning, word choice and names; keep the line layout. Source material still to prepare sits in `data/eval/<style>/raw/`.
1. `build_references --lt-public` - LanguageTool references through the public API into `data/eval/<style>/languagetool/`, with a pause of `LT_PUBLIC_DELAY_S` (10 s) after each text. Local mode needs Java and never waits. It also reports missing opus files.
2. `run_models` - generations through the app's task class and `src.llm.factory.get_model` (not `Task.run()`, whose random canary, truncation and trailer would change the prompt and output) -> `runs.jsonl`. The provider (`ollama` or `lmstudio`) is `config.PROVIDER`, override with `--provider`; default models come from `config.MODELS_BY_PROVIDER`.
3. `eval_errant`, `eval_embedding`, `eval_judge` - independent tests, any order
4. `aggregate` - `summary.csv` + `summary_long.csv`
5. `log_mlflow` - after all steps, publishes one MLflow run per model x setting to the experiment "Grammar Evaluation" at `MLFLOW_TRACKING_URI` (default `http://127.0.0.1:5001`, start it with `docker compose up -d mlflow`), in the format of `evaluation/notebooks/8_task_grammer_evaluation.ipynb`.

```sh
uv run python -m src.evaluation.build_references --style formal --prompt grammar --lt-public
uv run python -m src.evaluation.run_models       --style formal --prompt grammar
uv run python -m src.evaluation.eval_errant      --style formal --prompt grammar
uv run python -m src.evaluation.eval_embedding   --style formal --prompt grammar
uv run python -m src.evaluation.eval_judge       --style formal --prompt grammar
uv run python -m src.evaluation.aggregate        --style formal --prompt grammar
uv run python -m src.evaluation.log_mlflow       --style formal --prompt grammar
```

The gitignored files in `data/eval/results/<prompt>/<style>/` cache the steps, so an interrupted step resumes. Each (prompt, style) pair has its own cache, so formal and fiction runs never mix.

Deps: the eval deps (errant, language_tool_python, the `en_core_web_sm` wheel pinned in `[tool.uv.sources]`, mlflow) sit in the dev group of `pyproject.toml`; `uv sync` installs them. No anthropic, no `spacy download`.

## Prompt

Models get the app's `GrammarTask` prompt (`src/tasks/grammar.py`) as one user message, without the guardrail canary note, because its random canary would make every prompt different and break identical prompts across models. The RAG query is the app's: `retrieval_query` plus the first 2000 chars. `clean_output` strips the UNKNOWN WORDS trailer and wrapping triple quotes. Importing `src.tasks` adds about a minute of startup time.

## Data layout

All paths come from `common.EvalPaths(style, prompt)`.

```
data/eval/
  <style>/                         formal | fiction (also the Chroma collection name)
    raw/                           source files still to prepare (not read by any script)
    error_texts/<id>.txt           test texts with errors; the file name is the text id
    n_errors.json                  {"<id>": injected error count}, optional
    opus/<id>.txt                  hand-written reference correction
    languagetool/<id>.txt          LanguageTool reference (build_references)
    agreement.csv                  LT vs. opus edit overlap per text (eval_errant)
  prompts/<prompt>/                grammar | feedback | qna
    rag_cache.json                 frozen top-k guides per "<style>/<id>|k=<k>" (auto)
    judge_steps.json               frozen G-Eval steps per criterion (auto, hand-editable)
  results/<prompt>/<style>/        gitignored
    runs.jsonl                     one row per generation (shared input for all evals)
    errant.csv, embedding.csv, embedding_refs.csv, judge.jsonl, summary.csv, summary_long.csv
```

`runs.jsonl` row: `backend, model, setting_id, params, run, seed, text_id, style, prompt,
output, raw_output, wall_s, load_s, prompt_tokens, gen_tokens, gen_tok_per_s, done_reason`.
`done_reason = "length"` means the output hit the `NUM_PREDICT` (2048 token) cap and is cut off.
`backend` is the provider (`ollama` or `lmstudio`). For `lmstudio`, `load_s` and `gen_tok_per_s` are null (its API reports neither). `done_reason` is Ollama's `done_reason` or LM Studio's `finish_reason`; `length` means the cap was hit in both. `NUM_CTX` applies to Ollama only: LM Studio sets the context length when the model is loaded.
A separate vLLM runner only has to write the same fields (`backend="vllm"`).

Each MLflow run holds: params (prompt, model, temperature, top_p, style, collection, top_k, n_cases, n_runs, num_ctx, seed_base, judge and embed model, references), `prompt_template.txt`, `results.json` (per-generation table), step metrics per run, summary metrics, and the tags `testset_sha`, `style`, `prompt`.

## Modules

| Module | Purpose | Key functions |
|---|---|---|
| `config.py` | All settings: `EVAL_DATA_DIR`, `STYLES`, `PROMPTS`, `SUPPORTED_PROMPTS`, hosts, `PROVIDER`, `MODELS_BY_PROVIDER`, `SETTINGS` grid, `N_RUNS=5`, `runs_for`, `SEED_BASE`, `NUM_PREDICT`, `RAG_TOP_K=3`, embedding model, `EVAL_JUDGE_MODEL` (default the Nemotron hf.co tag), `MLFLOW_*` | `runs_for` |
| `common.py` | Eval paths, shared I/O, app prompt, RAG, Ollama calls, text normalization | `EvalPaths`, `eval_paths`, `add_target_args`/`paths_from_args`, `UnsupportedPrompt`, `load_testset`, `testset_sha`, `load_reference`, `load_runs`, `run_key`, `read_jsonl`/`append_jsonl`, `clean_output`, `normalize_for_scoring` (unwraps hard line breaks, collapses spaces, curly->straight quotes), `split_paragraphs`, `ollama_chat`, `ollama_embed`, `TASK_BY_PROMPT`, `retrieve_guides`, `build_messages` |
| `build_references.py` | LT reference (first suggestion per match, never overwrites hand-fixed files; `--lt-public`); reports missing hand-written opus files | `build_languagetool`, `report_missing` |
| `generation.py` | Builds the app's task with the factory model; invokes it directly to keep token counts and timings; per-provider options (Ollama `options`, LM Studio `max_tokens`/`seed`) | `build_task`, `generate`, `warm_up`, `Generation`, `UnknownProvider` |
| `run_models.py` | Model x setting x run x text loop; warm-up call so load time is excluded; seed = `SEED_BASE + run`; resumes via `run_key` | `main` (`--style`, `--prompt`, `--provider`, `--models`, `--runs`) |
| `eval_errant.py` | ERRANT edits extracted against the ORIGINAL text; edit key = `(o_start, o_end, c_str)` per paragraph; paragraph alignment, falls back to whole-text (`mode` column) if counts differ | `parse`/`edits` (lru-cached), `segment`, `f05`, `score_item`, `reference_agreement` |
| `eval_embedding.py` | Cosine similarity original vs output: `cos_doc` (mean-pooled), `cos_mean`/`cos_min` (paragraph pairs); calibration vs references in `embedding_refs.csv` | `embed` (cached, batched 32), `similarity` |
| `eval_judge.py` | G-Eval judge, sees only original + output. Criteria: correction (grammar + structure) and preservation (meaning + style), two calls per generation. Steps generated once and frozen. Score = logprob-weighted 1-5 if server returns logprobs, else median of `JUDGE_SAMPLES` samples. Refuses a judge that is in `MODELS` | `load_steps`, `score_prompt`, `weighted_score`, `score` (`--runs`, `--criteria`) |
| `aggregate.py` | Per run: sum ERRANT counts over texts (micro), then rates; then mean/median/std over runs, for one style and prompt | `errant_metrics`, `per_run_tables`, `summarize` (returns tables, writes nothing) |
| `log_mlflow.py` | Reads `results/` and `summarize()`, logs one MLflow run per model x setting; new runs on every call | `load_extras`, `log_group`, `main` |

## Metrics

### Shorthand

| Term | Meaning |
|---|---|
| hyp | hypothesis: the tested model's corrected text |
| ref, LT, opus | reference correction: `languagetool` (LT) or the hand-written `opus` file |
| edit | one ERRANT change from the original: the token span `(o_start, o_end)` and its replacement string. Two edits match only if both are identical |
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
| t / p | `temperature` / `top_p` (nucleus sampling: sample only from the smallest token set whose probability sums to p) |

### How values are aggregated

1. One value per (model, setting, run). A results folder holds one style and one prompt, so formal and fiction are never averaged together. ERRANT counts are summed over all texts of the run first, then turned into rates (micro average), so long texts weigh more. Embedding, judge and timing values are averaged over the texts.
2. `summary_long.csv` gives `mean`, `median`, `std` and `n_runs` of those per-run values for each metric. `summary.csv` holds the means only, one row per `model x setting_id`.
3. `std` is empty for `t0.0`, which runs once.

### summary.csv columns

| Column | Range | Better | Meaning |
|---|---|---|---|
| `tp_rate` | 0-1 | higher | **Primary.** `tp_spans / n_err_spans`: share of error spans (spans any reference edited) where the model made the same edit as a reference. Two references fixing one span differently count as one error |
| `tp_rate_consensus` | 0-1 | higher | `tp_cons / n_cons`: share of the edits both references agree on that the model also made. The most certain errors |
| `tp_rate_known` | 0-1+ | higher | `tp_spans / n_errors`, against the injected error count in the test set. Can exceed 1 when ERRANT splits one error into several edits. Only present if every text has `n_errors` |
| `precision_best` | 0-1 | higher | `tp_best / (tp_best + fp_best)`: share of model edits the best reference confirms |
| `f05_best` | 0-1 | higher | F0.5 from `tp_best`, `fp_best`, `fn_best` |
| `overcorr_per_100tok` | 0+ | lower | `100 * fp_union / orig_tokens`: model edits that no reference made, per 100 original tokens. Counts rewrites, added prefaces and hallucinated text |
| `cos_doc` | 0-1 | higher | cosine of the mean paragraph embedding of original vs. output (`EMBED_MODEL`). Measures content kept, not correctness. The references score 0.99-1.0 on `g_man_vs_computer`; outputs with a made-up or missing text drop below 0.8. An empty output scores 0 |
| `cos_mean` | 0-1 | higher | mean cosine of paragraph pairs (paragraph i vs. paragraph i). Only when output and original have the same paragraph count, else empty |
| `cos_min` | 0-1 | higher | lowest paragraph-pair cosine: catches one rewritten paragraph that the mean hides. Same condition as `cos_mean` |
| `judge_correction` | 1-5 | higher | LLM judge: spelling, punctuation, agreement and tense correct; awkward sentences rewritten |
| `judge_preservation` | 1-5 | higher | LLM judge: content unchanged, fine sentences left alone, voice, names and formatting kept |
| `wall_s` | s | lower | mean wall-clock seconds per generation (request sent to answer received, model already loaded) |
| `wall_s_median` | s | lower | median of the same, robust to one slow call |
| `gen_tok_per_s` | tok/s | higher | output tokens per second of pure generation (`eval_count / eval_duration`) |

Judge scores are the probability-weighted mean over the digits 1-5 when the server returns logprobs (`method = logprob_weighted` in `judge.jsonl`), else the median of `JUDGE_SAMPLES` samples (`sample_median`).

Known limitation: `cos_doc` averages all output paragraphs, so a preface such as "Certainly! Here is the revised text:" lowers it as if content had changed. On `g_man_vs_computer` good qwen2.5:0.5b outputs land at about 0.88 for this reason.

### Per-generation columns (errant.csv, MLflow `results.json`)

| Column | Meaning |
|---|---|
| `mode` | `paragraph` if output, original and references have the same paragraph count and are compared paragraph by paragraph, else `whole` (one block) |
| `orig_tokens` | spaCy tokens in the original |
| `n_hyp` | edits the model made |
| `tp_any` | model edits that match at least one reference |
| `tp_languagetool`, `tp_opus` | model edits that match that reference |
| `n_languagetool`, `n_opus` | edits that reference made |
| `n_err_spans`, `tp_spans` | error spans, and error spans the model fixed like a reference (see `tp_rate`) |
| `n_cons`, `tp_cons` | consensus edits, and those the model also made |
| `tp_best`, `fp_best`, `fn_best` | TP / FP / FN against the best reference per paragraph |
| `fp_union` | model edits no reference made |
| `n_errors` | injected error count from the test set |
| `tp_types` | ERRANT error types of the matched edits, e.g. `{"R:VERB:SVA": 3}` (replace, verb, subject-verb agreement) |

`data/references/agreement.csv` compares the references with each other: `n_lt`, `n_opus`, `n_cons` (edits both made) and `n_errors`. A low `n_cons` relative to `n_opus` means LanguageTool misses most errors, so `tp_rate_consensus` rests on few edits.

### MLflow

Experiment "Grammar Evaluation", one run per `model x setting_id`, named `<prompt>_<style>_<model>_<setting_id>`. Each call of `log_mlflow` adds new runs.

| Kind | Name | Meaning |
|---|---|---|
| tag | `testset_sha` | first 12 hex chars of a SHA-256 over the names and contents of `error_texts/*.txt`. Compare only runs with the same value |
| tag | `backend` | `ollama` or `vllm` |
| tag | `style`, `prompt` | the `--style` and `--prompt` of the run |
| param | `prompt` | the `--prompt` (app task whose template was used) |
| param | `provider`, `model` | backend and model tag |
| param | `temperature`, `top_p` | the setting |
| param | `style`, `collection` | prompt style(s) and Chroma collection(s) used |
| param | `top_k` | guide chunks retrieved per text |
| param | `n_cases`, `n_runs` | texts and runs in this MLflow run |
| param | `num_ctx`, `seed_base` | context window; run k uses seed `seed_base + k` |
| param | `judge_model`, `embed_model`, `references` | models and references used for scoring |
| artifact | `prompt_template.txt` | the app's `GrammarTask` template |
| artifact | `results.json` | one row per generation: `input` (original), `expected` (opus reference), `prompt` (full prompt sent), `response` (cleaned output), `error`, `latency_s`, `prompt_chars`, `response_chars`, `similarity`, `text_id`, `run`, `seed`, plus that generation's ERRANT, `cos_*` and `judge_*` values |
| step metric | `latency_s` | wall-clock seconds of generation i (step = generation index) |
| step metric | `similarity` | of generation i: difflib `SequenceMatcher` ratio between output and opus reference, 0-1, 1 = identical. Both are whitespace-normalized first, so reflowed lines don't count |
| step metric | `per_run/<metric>` | the per-run value of each summary.csv metric step = run index. Shows the spread across runs |
| summary | `<metric>`, `<metric>_std` | mean and std over runs of each summary.csv metric |
| summary | `error_rate` | share of generations with an empty output after cleaning |
| summary | `latency_mean_s`, `latency_max_s` | mean and max of `latency_s` |
| summary | `response_chars_mean` | mean output length in characters |
| summary | `similarity_mean` | mean of `similarity` |

`error_rate`, `latency_*`, `response_chars_mean` and `similarity` match the names in `evaluation/notebooks/8_task_grammer_evaluation.ipynb`, so both kinds of runs can be compared. The notebook computes `similarity` on raw strings, so its values run slightly lower.

## Decisions so far

- Judge model: NVIDIA Nemotron Nano 9B v2 (`hf.co/GGUF-A-Lot/NVIDIA-Nemotron-Nano-9B-v2-GGUF:Q4_K_M`), ~7 GB. Families under test (Granite, Qwen, Llama, Mistral, Gemma, Phi) are excluded as judges.
- Test models: granite4.2:8b, qwen3.5:9b, llama3.1:8b, mistral:7b-instruct (HF version), gemma4:e4b. Q&A (<=4B): gemma4:e4b, gemma4:e2b, phi4-mini:3.8b, gemma3:4b, qwen3.5:4b.
- Test data: 10 formal + 7 fiction texts (fiction: 1 per genre), errors injected with Claude. First item: `data/eval/formal/error_texts/g_man_vs_computer.txt` (30 injected + 6 existing errors, `n_errors=36`).

## Open TODOs

1. Optional: `eval_judge_calibration.py` - score ~30 outputs with Nemotron and Opus, Spearman >= ~0.7 means the local judge is usable.
2. Write the vLLM runner to the same `runs.jsonl` schema.
3. Tests for `clean_output` and `normalize_for_scoring`, written by a separate author.
