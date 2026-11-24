# Outline - Local Quill Report

The report presents Local Quill, a writing feedback system that runs entirely on the user's own machine. Section 1 introduces the privacy problem with cloud-hosted LLMs, the two target writing categories (academic essays and fictional prose) and the research question: can 8B-parameter local LLMs supported by RAG reach results comparable to GPT-5-nano, measured with G-Eval?

Section 2 covers the state of the art: the transformer architecture and the rise of LLMs, model compression through quantization, Ollama as a local runtime for quantized GGUF models, and retrieval-augmented generation. Section 3 describes the data: 12 publicly available PDF documents with essay writing guides, fiction writing guides and test materials such as Cambridge English sample responses.

Section 4 explains the methods, including the environment, prompt engineering for the grammar, clarity and style sub-tasks, the RAG pipeline and the G-Eval setup. Section 5 analyses the G-Eval results for the embedding and LLM models, and Section 6 discusses the conclusions and proposes further development.
