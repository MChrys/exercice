import io
import re
import os
import json
#import whisperx
#import torch
#import gc
import math
from collections import defaultdict

from opentelemetry import trace
from workflows.LLM_inf import ParallelLLMInference
from pathlib import Path
from omegaconf import DictConfig
import os
import docker

def get_ref_dicts(ref_file_path, epitran):
    """Function that creates 2 lookups for senators first names
    with values list of full names in clear text or its phonetic transcription 
    
    Keys are senators first names
    """
    span = trace.get_current_span()
    span.add_event("Starting creation of reference dictionaries")
    
    with io.open(ref_file_path, 'r') as f:
        content = f.read()
    senateurs = defaultdict(list)
    senateurs_ph = defaultdict(list)

    span.add_event("Processing senators' names")
    for senateur in content.split('\n'):
        first_name = senateur.split(' ')[0].lower()
        senateurs[first_name].append(senateur)
        senateurs_ph[first_name].append(epitran.transliterate(senateur.lower()))
    
    span.add_event("Finished creating reference dictionaries")
    return senateurs, senateurs_ph


def get_clean_name(text):
    """Simple function that strips person's title
    """
    span = trace.get_current_span()
    span.add_event("Cleaning person's name")
    
    pattern = re.compile(r'monsieur|m\.|mme|madame|président|présidente', re.I)
    s = re.sub(pattern, '', text)
    
    span.add_event("Finished cleaning person's name")
    return s.strip()

def correct_named_entity(ne_span, text):
    """Simple function that extends the span of Named Entity if '-' found at the end
    
    Parameters
    ----------
    ne_span : spacy.tokens.span.Span
        SpaCy ents
    text : str
        the text where ne_span was found
    """
    span = trace.get_current_span()
    span.add_event("Correcting named entity")
    
    if ne_span.end_char < len(text) and text[ne_span.end_char] == '-':
        result = ne_span.text + text[ne_span.end_char:].split(' ', 1)[0]
    else:
        result = ne_span.text
    
    span.add_event("Finished correcting named entity")
    return result


def spell_correct(transcription_list,
                   ref_file_path, 
                   epitran, 
                   nlp, 
                   jarowinkler, 
                   verbose=False):
    """ Funciton that corrects the spelling of Senators
8
    Parameters
    ----------
    transcription_list : str
        the text to be corrected
    ref_file_path: str
        path to senators reference file
    verbose: bool
        prints dict of spell corrections if set to True

    """
    span = trace.get_current_span()
    span.add_event("Starting spell correction")
    
    span.add_event("Getting reference dictionaries")
    senateurs, senateurs_ph = get_ref_dicts(ref_file_path,epitran)
    spello = {}
    text = ''
    #for transcription in transcription_list:
    #    text += transcription + '\n'
    corrected_list = transcription_list
    
    span.add_event("Processing text with NLP model")
    doc = nlp(corrected_list)
    
    span.add_event("Correcting named entities")
    for w in doc.ents:
        full_name = get_clean_name(correct_named_entity(w, text))
        if (w.label_ == 'PER' and len(full_name.split(' ')) > 1) :
            first_name, last_name = full_name.lower().split(' ', 1)
            first_name_ph = epitran.transliterate(first_name)
            if first_name in senateurs:
                last_name_ph = epitran.transliterate(last_name)
                score1 = [jarowinkler.similarity(last_name, re.sub(first_name, '', e.lower()).strip()) for e in senateurs[first_name]]
                score2 = [jarowinkler.similarity(last_name_ph, re.sub(first_name_ph, '', e).strip()) for e in senateurs_ph[first_name]]
                score = dict(map(lambda i,j : (i,j) , senateurs[first_name], zip(score1, score2)))
                score = sorted(score.items(), key=lambda item: sum(item[1]), reverse=True)
                # TODO: optimize thresholds for both JW scores wrt Precision
                if min(score[0][1]) > 0.7 and max(score[0][1]) > 0.8 and full_name.lower() != score[0][0].lower():
                    spello[last_name] = re.sub(first_name, '', score[0][0], flags=re.I)

    
    span.add_event("Printing corrections if verbose")
    if verbose:
        for k in spello:
            print('%s -> %s' %(k, spello[k]))

    span.add_event("Applying corrections to transcription list")
    for k in spello:
        pattern = re.compile(r'%s(?!\\w)' %k, re.I)
        for _, e in enumerate(corrected_list):
            e['text'] = re.sub(pattern, '%s' % spello[k], transcription_list[_]['text'])

    span.add_event("Spell correction completed")
    return corrected_list

def read_json_file(data_dir):
    for files in data_dir.glob('*.json'):
        with open(files, 'r', encoding='utf-8') as f:
            data = json.load(f)

    return data

# def transcribe(audio_file_path, model_name, device, language, compute_type, batch_size):
#     # Fonction pour transcrire un fichier audio en utilisant WhisperX
#     span = trace.get_current_span()
#     span.add_event("Starting transcription")
#     
#     # Options pour le modèle ASR
#     asr_options=dict(
#         max_new_tokens=128,
#         clip_timestamps="0",
#         hallucination_silence_threshold=0.2,
#         hotwords=[]
#     )
#     
#     # Chargement du modèle ASR
#     span.add_event("Loading ASR model")
#     model = whisperx.load_model(model_name, 
#                                 device, 
#                                 language=language, 
#                                 compute_type=compute_type, 
#                                 asr_options=asr_options)
#     
#     # Chargement du fichier audio
#     span.add_event("Loading audio file")
#     audio = whisperx.load_audio(audio_file_path)
#     
#     # Transcription de l'audio
#     span.add_event("Transcribing audio")
#     result = model.transcribe(audio, batch_size=batch_size)
#     
#     # Nettoyage de la mémoire
#     gc.collect()
#     torch.cuda.empty_cache()
#     del model
#     
#     # Chargement du modèle d'alignement
#     span.add_event("Loading alignment model")
#     model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
#     span.add_event("Aligning transcription")
#     result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
#     gc.collect()
#     torch.cuda.empty_cache()
#     del model_a
#     
#     # Chargement et exécution du modèle de diarisation
#     span.add_event("Loading diarization model")
#     diarize_model = whisperx.DiarizationPipeline(device=device)
#     span.add_event("Performing diarization")
#     diarize_segments = diarize_model(audio)
#     gc.collect()
#     torch.cuda.empty_cache()
#     del diarize_model
#     
#     # Attribution des locuteurs aux mots
#     span.add_event("Assigning speakers to words")
#     result = whisperx.assign_word_speakers(diarize_segments, result)
#     span.add_event("Transcription completed")
#     return result



def parse_whisperx_output(w_result):
    span = trace.get_current_span()
    span.add_event("Starting to parse WhisperX output")
    result = []
    for seg in w_result['segments']:

        try:
            if result and seg['speaker'] == result[-1]['speaker']:
                updated_seg = result[-1]
                updated_seg['text'] += seg['text']
                updated_seg['end'] = seg['end']
                result[-1] = updated_seg
            else:
                new_seg = {'start': seg['start'], 'end': seg['end'],
                           'speaker': seg['speaker'], 'text': seg['text']}
                result.append(new_seg)
        except KeyError as e:
            span.add_event(f"Key not found: {e}")

    span.add_event("Finished parsing WhisperX output")
    return result

def format_for_output(transcription_list):
    span = trace.get_current_span()
    span.add_event("Starting to format output")
    newline = '<br />'
    text = ""
    for e in transcription_list:
        if isinstance(e, dict):
            text += newline.join([e['speaker']+' :', e['text']]) + 2*newline
    span.add_event("Finished formatting output")
    return text


def normalize_and_parse_text(text):
    span = trace.get_current_span()
    span.add_event("Starting text normalization and parsing")
    text = re.sub(r'<//cr>|<\\cr>', '</cr>', text)
    pattern = re.compile(r'<cr>(.*?)(?=<cr>|</cr>|$)', re.DOTALL)
    matches = pattern.findall(text)
    
    filtered_matches = [match for match in matches if match.strip()]
    span.add_event("Finished text normalization and parsing")
    return filtered_matches

def apply_parse_and_reformat(transcription_list):
    span = trace.get_current_span()
    span.add_event("Starting apply_parse_and_reformat function")

    for transcription in transcription_list:
        span.add_event("Processing individual transcription")

        parsed_texts = normalize_and_parse_text(transcription['text'])

        if parsed_texts:
            span.add_event("Parsed texts found, joining and stripping")
            transcription['text'] = " ".join(parsed_texts).strip()
        else:
            span.add_event("No parsed texts found, setting empty string")
            transcription['text'] = ""

    span.add_event("Finished apply_parse_and_reformat function")
    return transcription_list

def transcribe_docker(audio_file_path, model_name, device, language, compute_type, batch_size):
    span = trace.get_current_span()
    span.add_event("Starting Docker transcription")
    
    client = docker.from_env()
    
    current_dir = os.path.abspath(os.path.dirname(__file__))

    span.add_event("Running Docker container for transcription")
    container = client.containers.run(
        'whisperx-transcriber',
        command=f"/data/{os.path.basename(audio_file_path)}",
        volumes={current_dir: {'bind': '/data', 'mode': 'ro'}},
        remove=True,
        stdout=True,
        stderr=True
    )

    output = container.decode('utf-8')
    try:
        span.add_event("Parsing transcription output")
        result = json.loads(output)
        span.add_event("Transcription completed successfully")
        return result
    except json.JSONDecodeError:
        span.add_event("Error in WhisperX output")
        print("Error in WhisperX output:", output)
        raise Exception("Transcription failed")
    


async def step_llm_inference(
                  input_text : list,
                  system_placeholder: str,
                  user_prompt_path :str,
                  config : DictConfig
                  )->None:
    span = trace.get_current_span()
    span.add_event("Starting LLM inference")

    c_llm_inference =  ParallelLLMInference( config.llm_api.api_url,
                                            config.llm_api.llm_model_name,
                                            config.llm_api.api_key, 
                                            config.llm_api.hf_model_name, 
                                            config.max_tokens, 
                                            config.max_concurrent_requests, 
                                            config.prompts.system, 
                                            system_placeholder, 
                                            user_prompt_path
                                            )
    span.add_event("Performing LLM inference")
    inflist = await c_llm_inference.LLM_inference(input_text)
    span.add_event("LLM inference completed")
    return format_for_output(apply_parse_and_reformat(inflist))


def transcribe_empty(file_path, percent_value=100,sub_path=None ):
    span = trace.get_current_span()
    span.add_event("Starting empty transcription")

    
    
    #current_dir = Path.cwd()
    #if current_dir.name == 'test':
    #    relative_path = current_dir.parent / 'data' / file_path
    #else : 
    #    relative_path = Path('data') / file_path
    relative_path= file_path
    
    try:
        span.add_event("Reading transcription file")
        with open(relative_path, 'r') as file:
            transcription_list = json.load(file)
    except FileNotFoundError as e:
        span.add_event(f"File not found: {relative_path}")
        span.record_exception(e)
        transcription_list = []
        raise
    except json.JSONDecodeError as e:
        span.add_event(f"JSON decoding error: {relative_path}")
        span.record_exception(e)
        transcription_list = []
        raise
    #segments = transcription_list['segments']
    #nb_elements = math.ceil(len(segments) * percent_value / 100)
    
    span.add_event("Sampling transcription list")
    #transcription_list['segments']=segments[:nb_elements]
    
    span.add_event("Empty transcription completed")
    return transcription_list