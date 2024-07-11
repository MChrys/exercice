
from LLM_inf import ParallelLLMInference

def llm_inference(c_transcription_list :list,
                  llm_model_name :str, 
                  api_key :str, 
                  hf_model_name :str, 
                  max_tokens :int,
                  max_concurrent_requests :int, 
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
    c_llm_enhanced_transcription_list = c_llm_inference.LLM_inference(c_transcription_list)
    c_verbatim_output = format_for_output(apply_parse_and_reformat(c_llm_enhanced_transcription_list))

    st.session_state["c_verbatim"] = c_verbatim_output