# Example usage

from workflows.nlp_steps import transcribe_empty, step_llm_inference, parse_whisperx_output
from conf import cfg
import asyncio
async def main():

    input = transcribe_empty("data/transcribe_encoded.json")
    input = parse_whisperx_output(input)
    result = await step_llm_inference(input,  cfg.placeholders.correction,             
                                              cfg.prompts.normalisation,
                                            cfg )
    return result

if __name__ == "__main__":
    import asyncio
    
    result = asyncio.run(main())
    print(result)