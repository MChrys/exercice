import asyncio
import logging
from workflows.pipeline import Pipeline, Step,Parameters
from workflows.nlp_steps import (transcribe_empty,
                       parse_whisperx_output, 
                       format_for_output, 
                       spell_correct, 
                       step_llm_inference)
from opentelemetry import trace
import spacy
import epitran
from similarity.jarowinkler import JaroWinkler
from conf import (cfg, 
                model_name, 
                device, 
                language, 
                compute_type, 
                batch_size,
                senators_file_path)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def func(val: int):
    span = trace.get_current_span()
    span.add_event(f"func called with value: {val}")
    result = val 
    span.add_event(f"func returns: {result}")
    return result

async def async_func(val: int):
    span = trace.get_current_span()
    span.add_event(f"async_func called with value: {val}")
    await asyncio.sleep(1)
    result = val 
    span.add_event(f"async_func returns: {result}")
    return result

async def main():
    jarowinkler = JaroWinkler()
    epi = epitran.Epitran(cfg["epi"])
    nlp = spacy.load(cfg["nlp"])
    pipe = Pipeline(cfg)
    
    step1 = Step(func)
    step2 = Step(async_func)
    step3 = Step(func)
    transcribe = Step(transcribe_empty)
    parse_transcription = Step(parse_whisperx_output)
    formatted_verbatim = Step(format_for_output)
    c_transcription_list = Step(spell_correct)
    c_transcription_list.set_params(Parameters(args = [senators_file_path, epi, nlp, jarowinkler]
                                                ,kwargs={"verbose":True}))
    
    c_verbatim_output = Step(step_llm_inference)
    c_verbatim_output.set_params(Parameters(args = [
                                                    cfg.placeholders.correction,             
                                                    cfg.prompts.normalisation,
                                                    cfg]))
    parsed_cri = Step(step_llm_inference)
    parsed_cri.set_params(Parameters(args = [
                                            cfg.placeholders.redaction,
                                            cfg.prompts.cri,
                                            cfg]))
    parsed_cra = Step(step_llm_inference)
    parsed_cra.set_params(Parameters(args = [
                                            cfg.placeholders.redaction,
                                            cfg.prompts.cra,
                                            cfg]))
    parsed_cred = Step(step_llm_inference)
    parsed_cred.set_params(Parameters(args = [
                                            cfg.placeholders.redaction,
                                            cfg.prompts.cred,
                                            cfg]))

    pipe >>  transcribe >> parse_transcription >> formatted_verbatim >> c_transcription_list 
    pipe | parse_transcription >> parsed_cri + parsed_cra + parsed_cred
    try:

        pipeline_future = asyncio.create_task(pipe.start("data/transcribe_encoded.json"))
        

        logger.info(f"step1 output: ======> {len(await transcribe.output)}")
        logger.info(f"step2 output: ======> {len(await parse_transcription.output)}")
        logger.info(f"step3 output: ======> {len(await formatted_verbatim.output)}")
        logger.info(f"step4 output: ======> {len(await c_transcription_list.output)}")
        
        

        result = await pipeline_future

    except asyncio.CancelledError:
        logger.info("Pipeline execution was cancelled")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
 
        pipeline_future.cancel()
    print()
if __name__ == "__main__":
    asyncio.run(main())