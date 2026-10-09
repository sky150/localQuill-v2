import os
import json
import time
from dotenv import load_dotenv
import src.evaluation.metrics_retrieval as metrics_retrieval
from src.tasks.utility import (
    get_db,
    text_normalization,
    similarity_search_eval,
)
from eval_config import EVAL_CONFIG, get_eval_results, save_eval_record


def retrieval_evaluation():
    with open("tests/eval/retrieval_test_questions.json", "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    print(f"=== Starting Retrieval Evaluation ===")
    print(f"Embedding Model: {EVAL_CONFIG['embedding_model']}")
    print(f"Chunk Size: {EVAL_CONFIG['chunk_size']}")
    print("=" * 36 + "\n")

    try:
        eval_db = get_db(
            chroma_path=EVAL_CONFIG["eval_db_path"],
            collection_name=EVAL_CONFIG["collection_name"],
        )
    except Exception as e:
        print(f"Could not load test database. Error: {e}")
        return

    results_log = []
    run_start = time.time()

    for i, tc in enumerate(test_cases):
        query = tc["query"]
        print(f"Test Case {i+1}/{len(test_cases)}: {query}")

        normalized_query = text_normalization(query)

        results = similarity_search_eval(
            db=eval_db,
            query=normalized_query,
            top_k=EVAL_CONFIG["top_k"],
        )

        if isinstance(results, str):
            print("No chunks retrieved for this query.")
            print("-" * 36)
            continue

        # auskommentiert weil wir nur retrieval testen wollen anstatt alle kategorien
        # focus_scores = {}
        # for focus, focus_results in results.items():
        # if not focus_results:
        #     continue

        results = similarity_search_eval(
            eval_db, normalized_query, top_k=EVAL_CONFIG["top_k"]
        )
        retrieved_chunks = [doc.page_content for doc, _ in results]
        retrieved_metadata = [doc.metadata for doc, _ in results]

        expected_concept = tc.get("expected_concept", "")
        expected_doc = tc.get("expected_document", "N/A")

        precision, precision_reason, recall, recall_reason = (
            metrics_retrieval.evaluate_retrieval(
                query, retrieved_chunks, expected_concept
            )
        )
        hit = metrics_retrieval.evaluate_document_hit(expected_doc, retrieved_metadata)

        print(f" Precision@K: {precision:.2f} | Recall@K: {recall:.2f} | Hit: {hit}")
        print(f"    P: {precision_reason}")
        print(f"    R: {recall_reason}")

        print("-" * 50)
        results_log.append(
            {
                "query": query,
                "expected_document": expected_doc,
                "expected_concept": expected_concept,
                "precision": precision,
                "precision_reason": precision_reason,
                "recall": recall,
                "recall_reason": recall_reason,
                "document_hit": hit,
            }
        )

    duration = time.time() - run_start

    if results_log:
        average_precision = sum(r["precision"] for r in results_log) / len(results_log)
        average_recall = sum(r["recall"] for r in results_log) / len(results_log)
        total_hits = sum(1 for r in results_log if r["document_hit"])
        average_hit_rate = total_hits / len(results_log)

        print(f"\nAverage Precision@K: {average_precision:.2f}")
        print(f"Average Recall@K:    {average_recall:.2f}")
        print(f"Average Hit Rate:    {average_hit_rate:.2f}")

        record = get_eval_results(
            "retrieval",
            duration,
            average_precision,
            average_recall,
            results_log,
            average_hit_rate,
        )
        save_eval_record(record)


if __name__ == "__main__":
    load_dotenv()
    retrieval_evaluation()
