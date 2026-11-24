# Research Notes - Local LLMs and RAG

The transformer architecture was introduced in the 2017 paper "Attention Is All You Need" by Vaswani et al. It replaced recurrence with self-attention, which lets the model relate every token to every other token in the input.

Quantization stores model weights with fewer bits, for example 4-bit or 8-bit integers instead of 16-bit floats. This reduces memory use and speeds up inference on consumer hardware, usually with a small loss in quality. Post-training quantization converts an already trained model, while quantization-aware training simulates the lower precision during training.

Retrieval-augmented generation (RAG) was described by Lewis et al. in 2020. A retriever finds relevant passages in an external document store, and these passages are added to the prompt so that the generator can base its answer on them.

G-Eval (Liu et al., 2023) uses a strong LLM as a judge. The judge receives evaluation criteria, generates evaluation steps through chain-of-thought, and then scores the output.
