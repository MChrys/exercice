import spacy
import epitran
from similarity.jarowinkler import JaroWinkler

from workflows.pipeline import Pipeline, Step,Parameters
from workflows.nlp_steps import (transcribe_empty,
                       parse_whisperx_output, 
                       format_for_output, 
                       spell_correct, 
                       step_llm_inference, 
                       parse_speaker_text)
from conf import (cfg, 
                senators_file_path)

jarowinkler = JaroWinkler()
epi = epitran.Epitran(cfg["epi"])
nlp = spacy.load(cfg["nlp"])
llm_pipe = Pipeline(cfg)


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
parse_inf_text = Step(parse_speaker_text)
parse_inf_text2 = Step(parse_speaker_text)

llm_pipe >>  transcribe >> parse_transcription >> formatted_verbatim >> c_transcription_list 
llm_pipe | c_transcription_list >> parse_inf_text >> c_verbatim_output
llm_pipe | c_verbatim_output >> parse_inf_text2 >> parsed_cri# + parsed_cra + parsed_cred
