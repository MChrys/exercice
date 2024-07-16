
import pathlib




import asyncio
from workflows.pipeline import Pipeline, Step,Parameters

#from conf  import cfg , senators_file_path

import spacy
import epitran
from similarity.jarowinkler import JaroWinkler

import io
import re
import os
import json

import math
from collections import defaultdict

from opentelemetry import trace
from workflows.LLM_inf import ParallelLLMInference
from pathlib import Path
from omegaconf import DictConfig
import os
import docker

import logging



from workflows.pipeline import Pipeline, Step,Parameters
from workflows.nlp_steps import (transcribe_empty,
                       parse_whisperx_output, 
                       format_for_output, 
                       spell_correct, 
                       step_llm_inference)

from conf import (cfg, 
                model_name, 
                device, 
                language, 
                compute_type, 
                batch_size,
                senators_file_path)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

jarowinkler = JaroWinkler()
epi = epitran.Epitran(cfg["epi"])
nlp = spacy.load(cfg["nlp"])

transcription = Step(transcribe_empty)

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

llm_workflow= Pipeline()


async def main():

    loop = asyncio.get_running_loop()

    llm_workflow >> transcription >> parse_transcription #>> formatted_verbatim # >> c_transcription_list 
    #llm_workflow | c_transcription_list >> c_verbatim_output + parsed_cri + parsed_cra + parsed_cred

    results = loop.create_task(llm_workflow.start("/Users/chrysostomebeltran/Downloads/exercise/data/transcribe_encoded.json"))
    trans = await transcription.output
    results = await results
    #spell_correct(results, senators_file_path, epi, nlp, jarowinkler, verbose=True)



if __name__ == "__main__":

    asyncio.run(main())