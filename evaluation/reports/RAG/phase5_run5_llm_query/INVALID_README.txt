The INVALID_* files are a stopped run 5 (2026-11-15) and must not be used for results.
Bug: the judging pass did not call use_variant(), so every cache lookup used the last
variant's settings (retrieval: no fixed guideline file). The lookups missed and new answers
were generated with the wrong settings while labelled hybrid/fixed. Evidence: hybrid formal
and fiction feedback share prompt hash aa3ba707, which is only possible without the fixed file.
Fixed in tests/eval/rag/run_task_eval.py. Runs 2-4 are not affected (no hash in the cache
key, or only one variant).
