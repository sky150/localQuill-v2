"""Evaluation settings. Edit this file, not the scripts.

Run every script from the repo root as a module, e.g.:
    uv run python -m src.evaluation.run_models
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths -------------------------------------------------------------------
# All eval data lives under data/eval. common.EvalPaths resolves the folders for one
# (style, prompt) pair; every script takes both as required --style / --prompt arguments.
REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DATA_DIR = REPO_ROOT / "data" / "eval"
STYLES = ("formal", "fiction")                # = data/eval/<style>/ and the Chroma collection name
PROMPTS = ("grammar", "feedback", "qna")      # = data/eval/prompts/<prompt>/
SUPPORTED_PROMPTS = PROMPTS
# Feedback runs on the formal texts only: it compares against last year's DISPRO1 run, whose
# fiction texts (LotR) were removed for licensing reasons.
STYLES_BY_PROMPT = {"grammar": STYLES, "feedback": ("formal",), "qna": STYLES}
REFERENCES = ("languagetool", "opus")         # = data/eval/<style>/<reference>/<id>.txt
# Feedback reads the original texts DISPRO1 used (raw/), the others the texts with injected errors.
INPUT_DIR_BY_PROMPT = {"grammar": "error_texts", "feedback": "raw", "qna": "error_texts"}

# Steps of run_all per prompt. Feedback and Q&A outputs are commentary and answers, which
# ERRANT and the embedding test cannot score, so only the judge scores them.
STEPS_BY_PROMPT = {
    "grammar": ("build_references", "run_models", "eval_errant", "eval_embedding", "eval_judge", "aggregate"),
    "feedback": ("run_models", "eval_judge", "aggregate"),
    "qna": ("build_qna_index", "run_models", "eval_judge", "aggregate"),
}

# --- Hosts -------------------------------------------------------------------
# Set OLLAMA_URL in .env, not here: the tested models are built by the app's factory, which
# reads OLLAMA_URL from src.config, so a URL hardcoded in this file would miss them.
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
JUDGE_URL = os.getenv("JUDGE_URL", OLLAMA_URL)    # desktop that hosts the judge
EMBED_URL = os.getenv("EMBED_URL", "http://localhost:11434")
REQUEST_TIMEOUT_S = 1800

# --- Models under test -------------------------------------------------------
# The tested models run through the app's model factory (src.llm.factory.get_model), so the
# server URLs come from the app config (OLLAMA_URL, LMSTUDIO_URL in .env). The eval provider is
# hardcoded here on purpose: .env's LLM_PROVIDER configures the app, not the experiment.
PROVIDERS = ("ollama", "lmstudio")
PROVIDER = "ollama"


# Models under test per prompt, then per provider (Ollama tag vs. LM Studio model id).
# Pipeline tests: qwen2.5:0.5b (Ollama), passed with --models.
MODELS_BY_PROMPT = {
    "grammar": {
        "ollama": ["granite4.2:8b", "qwen3.5:9b", "llama3.1:8b", "mistral:7b-instruct", "gemma4:e4b"],
        "lmstudio": ["qwen2.5-coder-0.5b-instruct"],
    },
    # Up to 4B parameters: Q&A runs next to the larger grammar model in the app.
    "qna": {
        "ollama": ["gemma4:e2b", "gemma4:e4b", "phi4-mini:3.8b", "gemma3:4b", "qwen3.5:4b"],
        #"ollama": ["gemma4:e2b"],
        "lmstudio": ["qwen2.5-coder-0.5b-instruct"],
    },
    # Best feedback model of last semester, rerun with DISPRO1's settings.
    "feedback": {
        "ollama": ["ministral-3:8b"],
        "lmstudio": ["qwen2.5-coder-0.5b-instruct"],
    },
}


def models_for(prompt: str, provider: str = PROVIDER) -> list[str]:
    return MODELS_BY_PROMPT[prompt][provider]


# Each setting is one metaparameter combination. top_p stays 1.0 (off), which leaves temperature
# as the only variable. Every setting runs N_RUNS times, except temperature 0 (see runs_for).
SETTINGS_BY_PROMPT = {
    # Grammar correction wants near-deterministic output, so only temperatures 0 to 0.3.
    "grammar": [
        {"id": "t0.0", "temperature": 0.0, "top_p": 1.0},
        {"id": "t0.1", "temperature": 0.1, "top_p": 1.0},
        {"id": "t0.2", "temperature": 0.2, "top_p": 1.0},
        {"id": "t0.3", "temperature": 0.3, "top_p": 1.0},
    ],
    # Answers may be phrased freely, so Q&A tests a wider range.
    "qna": [
        {"id": "t0.0", "temperature": 0.0, "top_p": 1.0},
        {"id": "t0.2", "temperature": 0.2, "top_p": 1.0},
        {"id": "t0.7", "temperature": 0.7, "top_p": 1.0},
        {"id": "t1.0", "temperature": 1.0, "top_p": 1.0},
    ],
    # DISPRO1 winnign model was at temperature 0.1.
    "feedback": [
        {"id": "t0.1", "temperature": 0.1, "top_p": 1.0},
    ],
}
# One context size for every tested model and both providers. Ollama takes it per request;
# LM Studio fixes it at load time, so run_models (re)loads the model with it (generation.py).
NUM_CTX = int(os.getenv("LLM_CONTEXT", "16384"))
# Small models can loop without stopping; uncapped, qwen2.5:0.5b on CPU ran past the
# 1800 s request timeout on one ~400-token text. Rows with done_reason "length" hit the cap.
NUM_PREDICT = 2048
N_RUNS = 3                 # 3-5 retries per setting


def runs_for(setting: dict, n_runs: int = N_RUNS) -> int:
    """Temperature 0 is greedy decoding: seed and top_p have no effect and every run gives
    the same output, so one run is enough. Its std over runs is therefore empty (NaN)."""
    return 1 if setting["temperature"] == 0 else n_runs

SEED_BASE = 1000           # run k uses seed SEED_BASE + k, so runs differ but are reproducible

# --- RAG (writing guides) ----------------------------------------------------
RAG_TOP_K = 3              # hardcoded until the embedding-test results are in
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma")
RAG_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")     # must be the model the collections were built with
RAG_QUERY_CHARS = 2000     # query = first N chars of the input text

# --- Q&A documents -------------------------------------------------------------
# Every text gets its own userdocs and outline collections in an eval-only Chroma, so the app's
# own userdocs never mix with test documents (build_qna_index).
QNA_CHROMA_PATH = os.getenv("QNA_CHROMA_PATH", str(EVAL_DATA_DIR / "chroma_qna"))
QNA_MAX_USERDOCS = 3
# QATask clips the attached document to 4000 chars (src/tasks/qa.py). The eval keeps that limit,
# since the app is under test, so answer-sheet facts must come from the first 4000 chars.
QNA_DOC_CHARS = 4000
QNA_USERDOCS_CHARS = 3000  # = QATask.build_prompt's clip of USER DOCUMENTS
QNA_OUTLINE_CHARS = 800    # = QATask.build_prompt's clip of USER OUTLINE

# --- References --------------------------------------------------------------
LT_LANGUAGE = "en-GB"      # the app's grammar prompt says "Write in en-GB"
# The public API is rate-limited per IP and blocks bursts; the pause after each text keeps a
# full build under its limit. Local mode (Java) has no limit and never waits.
LT_PUBLIC_DELAY_S = 10

# --- Embedding similarity test -----------------------------------------------
EMBED_MODEL = os.getenv("EVAL_EMBED_MODEL", "embeddinggemma-2:740m")   # set to your Gemma4 740 MB embedding tag
EMBED_PREFIX = ""          # e.g. "task: sentence similarity | query: " if the model expects task prefixes

# --- LLM judge (G-Eval) ------------------------------------------------------
# EVAL_JUDGE_MODEL, not JUDGE_MODEL: .env's JUDGE_MODEL is read by the old deepeval
# scripts and is not a valid Ollama tag.
JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "hf.co/GGUF-A-Lot/NVIDIA-Nemotron-Nano-9B-v2-GGUF:Q4_K_M")   # never one of tested MODELS
# Feedback keeps DISPRO1's judge, so its scores compare with last year's directly.
JUDGE_MODEL_BY_PROMPT = {"grammar": JUDGE_MODEL, "qna": JUDGE_MODEL,
                         "feedback": os.getenv("EVAL_FEEDBACK_JUDGE_MODEL", "llama3.1:8b")}
JUDGE_SAMPLES = 3          # used only when the judge returns no logprobs
JUDGE_TEMPERATURE = 0.7    # for sampling fallback
JUDGE_NUM_CTX = 16384

# --- MLflow (published after all eval steps) ---------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
MLFLOW_EXPERIMENT_BY_PROMPT = {"grammar": "Grammar Evaluation", "qna": "Q&A Evaluation",
                               "feedback": "Feedback Evaluation"}

# --- DISPRO1 comparison (feedback) ---------------------------------------------
DISPRO1_RESULTS = REPO_ROOT / "evaluation" / "reports" / "DISPRO1" / "eval_generation_essay_results.jsonl"
