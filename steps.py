
from LLM_inf import ParallelLLMInference

async def  llm_inference(
                  system_prompt_path :str, 
                  system_placeholder: str,
                  user_prompt_path :str)->None:
    c_llm_inference = ParallelLLMInference(llm_model_name,
                                            api_key, 
                                            hf_model_name, 
                                            max_tokens, 
                                            max_concurrent_requests, 
                                            "services/system_prompt.txt", 
                                            "la correction orthographique", 
                                            "services/prompt_normalisation_v0.txt"
                                            )
    inflist = c_llm_inference.LLM_inference(c_transcription_list)
    return format_for_output(apply_parse_and_reformat(inflist))

