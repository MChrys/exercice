import spacy
import epitran
from similarity.jarowinkler import JaroWinkler

from workflows.pipeline import Pipeline, Step,Parameters
from workflows.nlp_steps import (transcribe,
                       parse_whisperx_output, 
                       format_for_output, 
                       spell_correct, 
                       step_llm_inference)
from .. import (cfg, 
                model_name, 
                device, 
                language, 
                compute_type, 
                batch_size,
                senators_file_path,
                epi,
                nlp,
                jarowinkler)

jarowinkler = JaroWinkler()
epi = epitran.Epitran(cfg["epi"])
nlp = spacy.load(cfg["nlp"])

transcription = Step(transcribe)
transcription.set_params(Parameters(args = [model_name, device, language, compute_type, batch_size]))
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

llm_workflow >> transcription >> parse_transcription >> formatted_verbatim >> c_transcription_list 
llm_workflow | c_transcription_list >> c_verbatim_output >> parsed_cri + parsed_cra + parsed_cred