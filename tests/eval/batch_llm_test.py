import os
from run_generation_eval import run_full_evaluation_with_config


def test_run_full_evaluation(test_models: list, test_text_dict: dict, provider="ollama", style="formal",
                             result_file_name="eval_generation_essay_results.jsonl"):
    """Test the full evaluation process for grammar, style, and clarity."""
    
    total_runs = len(test_models) * len(test_text_dict)
    counter = 1
    print(f"Starting batch evaluation: {total_runs} runs ({len(test_models)} models x {len(test_text_dict)} texts)\n")

    for model in test_models:        
        print(f"\n{'='*50}")
        print(f"Testing model: {model}")
        print(f"{'='*50}")


        for key, value in test_text_dict.items():
            print(f"Evaluation {counter}/{total_runs} - Testing text: {key}")
            counter += 1
            
            try:
                # run_full_evaluation(value, style=style)
                run_full_evaluation_with_config(value, provider=provider, style=style, llm_model=model, 
                                                result_file_name=result_file_name, file_name=key)
            except Exception as e:
                print(f"ERROR with {model} on {key}: {e}")
                continue  # Continue to next text even if this one fails




def get_essay_dict():
    """Gather all the text files for Essays"""
    essay_folder = "data/eval/texts/test_essay/"
    essay_files = [f for f in os.listdir(essay_folder) if f.endswith(".txt")]
    essay_dict = {}

    for file in essay_files:
        file_path = os.path.join(essay_folder, file)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            essay_dict[file] = content

    return essay_dict

def get_fiction_dict():
    """Gather all the text files for Fiction"""
    fiction_folder = "data/eval/texts/test_fiction/"
    fiction_files = [f for f in os.listdir(fiction_folder) if f.endswith(".txt")]
    fiction_dict = {}

    for file in fiction_files:
        file_path = os.path.join(fiction_folder, file)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            fiction_dict[file] = content
            
    return fiction_dict

if __name__ == "__main__":
    """This will run the full evaluation process for multiple models and multiple texts in a batch manner and add them to the eval_result.json"""
    #####
    # The Following 3 Runs need to be strictly called by hand. 
    # They are batch calls, and will make countless LLM calls, so they run for ages.
    # Uncomment only one at a time
    #####
    test_models = [
        "llama3.1:8b",
        "mistral:7b", # 7b 
        "ministral-3:8b",
        # "mistral-nemo:12b", 
        "qwen2.5:7b",
        "qwen3:8b",
        "qwen3.5:9b",  # Too slow gets stuck for fiction, sometimes doesn't answer. Gets errors with empty response. Remove from final models
        # "qwen2.5:14b",
        # "qwen3:14b",
        # "mistral-small3.2", # 24b 
        # "qwen3.6:27b",  # Too slow gets stuck for fiction       
    ] 
    essays = get_essay_dict()
    result_file_name = "eval_generation_essay_results.jsonl"
    test_run_full_evaluation(test_models, essays, style="formal", result_file_name=result_file_name)
    

    test_models = [
        "qwen3:8b",
        "llama3.1:8b",
        "mistral:7b", # 7b 
        "ministral-3:8b",
        # "mistral-nemo:12b", 
        "qwen2.5:7b",
        # "qwen2.5:14b",  
        # "qwen3:14b",        
    ] 
    fiction = get_fiction_dict()
    fiction.pop("LotR_Chapter_1_2_1000.txt")    # Only the shortest is actually used. The others are overwhelmed during g-eval
    fiction.pop("LotR_Chapter_1_3_2000.txt")
    fiction.pop("LotR_Chapter_1_4_4000.txt")
    fiction.pop("LotR_Chapter_1_Full.txt")
    result_file_name = "eval_generation_fiction_results.jsonl"
    # test_run_full_evaluation(test_models, fiction, style="fiction", result_file_name=result_file_name)
    
    # OpenAI Comparison
    # test_models = ["gpt-5-nano"]    # Model overrides what stands in the .env file
    # test_run_full_evaluation(test_models, essays, provider="openai", style="formal", result_file_name=result_file_name)
    # test_run_full_evaluation(test_models, fiction, provider="openai", style="fiction", result_file_name=result_file_name)
    
    
    
    ###
    # Report
    # The qwen3:9b model. Will be cut out. Even if it performs better. It is not consitent and prone to get stuck.
    ###