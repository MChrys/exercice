import io
import re
import os
import json
import whisperx
import torch
import gc

from collections import defaultdict


from LLM_inf import ParallelLLMInference

from omegaconf import DictConfig
import os
import docker

def get_ref_dicts(ref_file_path ,epitran):
    """Function that creates 2 lookups for senators first names
    with values list of full names in clear text or its phonetic transcription 
    
    Keys are senators first names
    """
    with io.open(ref_file_path, 'r') as f:
        content = f.read()
    senateurs = defaultdict(list)
    senateurs_ph = defaultdict(list)

    for senateur in content.split('\n'):
        first_name = senateur.split(' ')[0].lower()
        senateurs[first_name].append(senateur)
        senateurs_ph[first_name].append(epitran.transliterate(senateur.lower()))
    return senateurs, senateurs_ph


def get_clean_name(text):
    """Simple function that strips person's title
    """
    pattern = re.compile(r'monsieur|m\.|mme|madame|président|présidente', re.I)
    s = re.sub(pattern, '', text)
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
    if text[ne_span.end_char] == '-':
        return ne_span.text + text[ne_span.end_char:].split(' ', 1)[0]
    else:
        return ne_span.text


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
    senateurs, senateurs_ph = get_ref_dicts(ref_file_path)
    spello = {}
    text = ''
    for transcription in transcription_list:
        text += transcription["text"] + '\n'
    corrected_list = transcription_list
    doc = nlp(text)
    for w in doc.ents:
        full_name = get_clean_name(correct_named_entity(w, text))
        if w.label_ == 'PER' and len(full_name.split(' ')) > 1:
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
    if verbose:

       # print('Corrections:')
        for k in spello:
            print('%s -> %s' %(k, spello[k]))

    for k in spello:
        pattern = re.compile(r'%s(?!\\w)' %k, re.I)
        for _, e in enumerate(corrected_list):
            e['text'] = re.sub(pattern, '%s' % spello[k], transcription_list[_]['text'])

    return corrected_list

def read_json_file(data_dir):
    for files in data_dir.glob('*.json'):
        with open(files, 'r', encoding='utf-8') as f:
            data = json.load(f)

    return data

def transcribe(audio_file_path, model_name, device, language, compute_type, batch_size):
    asr_options=dict(
        max_new_tokens=128,
        clip_timestamps="0",
        hallucination_silence_threshold=0.2,
        hotwords=[]
    )
    model = whisperx.load_model(model_name, 
                                device, 
                                language=language, 
                                compute_type=compute_type, 
                                asr_options=asr_options)
    audio = whisperx.load_audio(audio_file_path)
    result = model.transcribe(audio, batch_size=batch_size)
    gc.collect()
    torch.cuda.empty_cache()
    del model
    model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    gc.collect()
    torch.cuda.empty_cache()
    del model_a
    diarize_model = whisperx.DiarizationPipeline( device=device)
    diarize_segments = diarize_model(audio)
    gc.collect()
    torch.cuda.empty_cache()
    del diarize_model
    result = whisperx.assign_word_speakers(diarize_segments, result)
    return result



def parse_whisperx_output(w_result):
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
            print('key not found ', e)
    return result

def format_for_output(transcription_list):
    newline = '<br />'
    text = ""
    for e in transcription_list:
       # print('--------------------')
       # print(e)
        if isinstance(e, dict):
            text += newline.join([e['speaker']+' :', e['text']]) + 2*newline
    return text


def normalize_and_parse_text(text):
    text = re.sub(r'<//cr>|<\\cr>', '</cr>', text)
    pattern = re.compile(r'<cr>(.*?)(?=<cr>|</cr>|$)', re.DOTALL)
    matches = pattern.findall(text)
    
    filtered_matches = [match for match in matches if match.strip()]

    return filtered_matches

def apply_parse_and_reformat(transcription_list):

    for transcription in transcription_list:

        parsed_texts = normalize_and_parse_text(transcription['text'])

        if parsed_texts:

            transcription['text'] = " ".join(parsed_texts).strip()
        else:
            transcription['text'] = ""

    return transcription_list

def transcribe_docker(audio_file_path, model_name, device, language, compute_type, batch_size):
    client = docker.from_env()
    
    current_dir = os.path.abspath(os.path.dirname(__file__))

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
        result = json.loads(output)
        return result
    except json.JSONDecodeError:
        print("Error in WhisperX output:", output)
        raise Exception("Transcription failed")
    


async def  step_llm_inference(
                  input_text : list,
                  system_placeholder: str,
                  user_prompt_path :str,
                  config : DictConfig
                  )->None:

    c_llm_inference = ParallelLLMInference(config.llm_model_name,
                                            config.api_key, 
                                            config.hf_model_name, 
                                            config.max_tokens, 
                                            config.max_concurrent_requests, 
                                            config.prompts.system, 
                                            system_placeholder, 
                                            user_prompt_path
                                            )
    inflist = await c_llm_inference.LLM_inference(input_text)
    return format_for_output(apply_parse_and_reformat(inflist))