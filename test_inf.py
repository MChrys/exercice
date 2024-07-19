# Example usage
from workflows.LLM_inf import ParallelLLMInference
from workflows.nlp_steps import transcribe_empty, step_llm_inference
from config import initialize, compose
async def main():
    input = transcribe_empty("data/transcribe_encoded.json")
    result = await step_llm_inference(input, inference)
    return result

if __name__ == "__main__":
    initialize(config_path="config")
    cfg = compose(config_name="local")

    api_key = cfg["llm_api"]["api_key"]
    llm_model_name = cfg["llm_api"]["llm_model_name"]
    base_url = cfg["llm_api"]["api_url"]

    senators_file_path = cfg["senators_file_path"]

    hf_model_name = cfg["llm_api"]["hf_model_name"]
    max_tokens = cfg["max_tokens"]
    max_concurrent_requests = cfg["max_concurrent_requests"]

    exaion_model_name = llm_model_name
    exaion_api_key = api_key
   
    #hf_token = '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'

    system_prompt_path = "services/system_prompt.txt"
    user_prompt_path = "services/prompt_normalisation_v0.txt"
    #normalisation_prompt_path = "services/prompt_normalisation_v0.txt"
    system_placeholder = "rédaction de compte rendu"

    inference = ParallelLLMInference(base_url, 
                                     exaion_model_name, 
                                     exaion_api_key, hf_model_name, 
                                     max_tokens, max_concurrent_requests, 
                                     system_prompt_path, 
                                     system_placeholder, 
                                     user_prompt_path,   
                                     )
    
    # Example segments to infer
    segments = [
        {"speaker": "Speaker 1", "text": "Hello, this is a test."},
        {"speaker": "Speaker 2", "text": "Hi, this is another test."}
    ]

    
    results = inference.LLM_inference(segments)
    print("results : ")
    print(results)