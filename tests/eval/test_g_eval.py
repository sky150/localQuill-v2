import time
import src.evaluation.metrics_generation as metrics_generation
from src.model_query import query_rag
from eval_config import EVAL_CONFIG, get_eval_results, save_eval_record


def run_full_evaluation(user_prompt, style="formal"):
    """Run the full evaluation process for grammar, style, and clarity."""
    feedback = query_rag(
        user_prompt,
        style=style, 
        return_dict=True
    )

    print(f"=== Starting Generation Evaluation ===")
    print(f"LLM Model: {EVAL_CONFIG['llm_model']}")
    print("=" * 36 + "\n")

    run_start = time.time()
    results_log = []

    feedback = query_rag(user_prompt, style="formal", return_dict=True)  # fantasy

    grammar_feedback = "\n".join(feedback["grammar"])
    style_feedback = "\n".join(feedback["style"])
    clarity_feedback = "\n".join(feedback["clarity"])

    print("\n--- Running Grammar Evaluation ---")

    g_score, g_reason = metrics_generation.evaluate_grammar(
        user_prompt, grammar_feedback
    )
    results_log.append({"metric": "grammar", "score": g_score, "reason": g_reason})

    print("\n--- Running Style Evaluation ---")

    s_score, s_reason = metrics_generation.evaluate_style(user_prompt, style_feedback)
    results_log.append({"metric": "style", "score": s_score, "reason": s_reason})

    print("\n--- Running Clarity Evaluation ---")

    c_score, c_reason = metrics_generation.evaluate_clarity(
        user_prompt, clarity_feedback
    )
    results_log.append({"metric": "clarity", "score": c_score, "reason": c_reason})

    duration = time.time() - run_start
    average_score = (g_score + s_score + c_score) / 3

    print(f"\nAverage generation score: {average_score:.2f}")

    record = get_eval_results("generation", duration, average_score, results_log)
    save_eval_record(record)


if __name__ == "__main__":

    style="formal" # "formal" or "fiction"

    test_user_prompt = """It cannot be denied that tourism is an economic force \n
and that over the past decades it has gone from strength
to strength. Both texts give us some insight into one
important link for the industry, and that is the internet.
The first text makes it clear that travel, whether doing
so on a restricted budget or with no such limitations, is
facilitated with the amount of online material to choose
from. This may, of course, include such things as travel
vlogs, such as the ones produced by the author of the
second article. This second author writes from a personal
viewpoint and her perspective confirms that the world
wide web is indeed a rich resource.\n
\n
Reading between the lines, the travel vlogger appears
to show her passion and extraordinary enthusiasm for
travel. This is somewhat missing in the first text where
the process of identifying where to go and what to do is
made to sound so much more mechanical. The vlogger
thinks beyond what is in front of them and considers
deeply what the experiences bring her in terms of the
things she thinks are most important and valuable. The
writer of the first article almost dismisses this aspect by
focusing on the simplicity of taking a trip.
Although I have not travelled extensively, I can relate
to the vlogger and feel any tourism company would
benefit from the clear delight behind each experience in
promoting their company."""

    run_full_evaluation(test_user_prompt, style)
