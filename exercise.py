import io
import re
import streamlit as st
import pathlib
import os
import spacy
import epitran
import json
import streamlit_authenticator as stauth
import yaml
import whisperx
import torch
import gc
from yaml.loader import SafeLoader
from collections import defaultdict
from similarity.jarowinkler import JaroWinkler
from pydub import AudioSegment
from LLM_inf import ParallelLLMInference


app_dir = pathlib.Path(__file__).parent.absolute()
data_dir = app_dir / "data"
data_dir.mkdir(exist_ok=True)

device = "cuda"
batch_size = 16
compute_type = "float16"
model_name = "large-v2"
hf_token = os.getenv('HF_TOKEN')
senators_file_path = 'services/senateurs_name_last.txt'


llm_model_name = 'meta-llama-3-8b-instruct'
api_key = os.getenv('API_KEY')
hf_model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'

max_tokens = 3500
max_concurrent_requests = 10


jarowinkler = JaroWinkler()
epi = epitran.Epitran('fra-Latn-p')
nlp = spacy.load('fr_core_news_lg')


st.set_page_config(layout="wide")
st.session_state["verbatim"] = ''
st.session_state["c_verbatim"] = ''
st.session_state["c_cri"] = ''
st.session_state["c_cra"] = ''
st.session_state["c_cred"] = ''

st.markdown("<h1 style='text-align: center; color: grey;'>Outil d'aide à la génération du CRI, CRED et CRA</h1>", unsafe_allow_html=True)

# Authentification
with open('credentials.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)


def get_ref_dicts(ref_file_path):
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
        senateurs_ph[first_name].append(epi.transliterate(senateur.lower()))
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


def spell_correct(transcription_list, ref_file_path, verbose=False):
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
            first_name_ph = epi.transliterate(first_name)
            if first_name in senateurs:
                last_name_ph = epi.transliterate(last_name)
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

def transcribe(audio_file_path):
    model = whisperx.load_model(model_name, device, language='fr', compute_type=compute_type)
    audio = whisperx.load_audio(audio_file_path)
    result = model.transcribe(audio, batch_size=batch_size)
    gc.collect()
    torch.cuda.empty_cache()
    del model
    model_a, metadata = whisperx.load_align_model(language_code='fr', device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    gc.collect()
    torch.cuda.empty_cache()
    del model_a
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
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

def main():
    # File uploader
    audio_path = None
    percent_complete = 0
    col1, col2 = st.columns([0.3, 0.7], gap="medium", vertical_alignment="center")
    with col1:
        audio_file = st.file_uploader("Télécharger le fichier audio", type=["mp3", "wav"])
        audio_path = None
        if audio_file is not None:
            data_dir = pathlib.Path(__file__).parent / "data"
            data_dir.mkdir(exist_ok=True)
            audio_path = data_dir / audio_file.name
            with open(audio_path, 'wb') as f:
                f.write(audio_file.getvalue())
            if 'mp3' in audio_file.name:
                sound = AudioSegment.from_mp3(audio_path)
                audio_path_wav = audio_path.with_suffix('.wav')
                sound.export(audio_path_wav, format="wav", parameters=["-ar", "16000", "-ac", "1", "-ab", "32k"])
                audio_path = audio_path_wav
            w_result = transcribe(audio_path)
            transcription_list = parse_whisperx_output(w_result)

            # Appliquer parse_text_between_cr_tags et reformater
            formatted_verbatim = format_for_output(transcription_list)
            st.session_state["verbatim"] = formatted_verbatim

            c_transcription_list = spell_correct(transcription_list, senators_file_path, True)

            c_llm_inference = ParallelLLMInference(llm_model_name,
                                                   api_key, 
                                                   hf_model_name, 
                                                   hf_token, 
                                                   max_tokens, 
                                                   max_concurrent_requests, 
                                                   "services/system_prompt.txt", 
                                                   "la correction orthographique", 
                                                   "services/prompt_normalisation_v0.txt"
                                                   )
            c_llm_enhanced_transcription_list = c_llm_inference.LLM_inference(c_transcription_list)
            c_verbatim_output = format_for_output(apply_parse_and_reformat(c_llm_enhanced_transcription_list))

            st.session_state["c_verbatim"] = c_verbatim_output

            cri_llm_inferance = ParallelLLMInference(llm_model_name, 
                                                     api_key, 
                                                     hf_model_name, 
                                                     hf_token, 
                                                     max_tokens, 
                                                     max_concurrent_requests, 
                                                     "services/system_prompt.txt", 
                                                     "la rédaction de compte rendu", 
                                                     "services/CRI_prompt.txt"
                                                     )
            cri_llm_transcription_list = cri_llm_inferance.LLM_inference(c_transcription_list)
            parsed_cri = format_for_output(apply_parse_and_reformat(cri_llm_transcription_list))
            st.session_state["c_cri"] = parsed_cri

            cra_llm_inferance = ParallelLLMInference(llm_model_name, 
                                                     api_key, 
                                                     hf_model_name, 
                                                     hf_token, 
                                                     max_tokens, 
                                                     max_concurrent_requests, 
                                                     "services/system_prompt.txt", 
                                                     "la rédaction de compte rendu", 
                                                     "services/CRA_prompt.txt"
                                                     )
            cra_llm_transcription_list = cra_llm_inferance.LLM_inference(c_transcription_list)
            parsed_cra = format_for_output(apply_parse_and_reformat(cra_llm_transcription_list))
            st.session_state["c_cra"] = parsed_cra

            cred_llm_inferance = ParallelLLMInference(llm_model_name, 
                                                      api_key, 
                                                      hf_model_name, 
                                                      hf_token, 
                                                      max_tokens, 
                                                      max_concurrent_requests, 
                                                      "services/system_prompt.txt", 
                                                      "la rédaction de compte rendu", 
                                                      "services/CRED_prompt.txt"
                                                      )
            cred_llm_transcription_list = cred_llm_inferance.LLM_inference(c_transcription_list)
            parsed_cred = format_for_output(apply_parse_and_reformat(cred_llm_transcription_list))
            st.session_state["c_cred"] = parsed_cred
    
    if audio_path :
        with col2:
            st.audio(str(audio_path), format='audio/wav')

    col1, col2 = st.columns([0.5, 0.5], gap="medium")
    with col1:
        verbatim, correct_verbatim = st.tabs(["Verbatim", "Verbatim optimisé"])
        with verbatim:
            with st.container(height=700):
                st.markdown(st.session_state["verbatim"], unsafe_allow_html=True)
        with correct_verbatim:
            with st.container(height=700):
                st.markdown(st.session_state["c_verbatim"], unsafe_allow_html=True)

    with col2:
        tab1, tab2, tab3 = st.tabs(["  CRI  ", "  CRA  ", "  CRED  "])
        with tab1:
            with st.container(height=700):
                st.markdown( st.session_state["c_cri"], unsafe_allow_html=True)
        with tab2:
            with st.container(height=700):
                st.markdown( st.session_state["c_cra"], unsafe_allow_html=True)
        with tab3:
            with st.container(height=700):
                st.markdown( st.session_state["c_cred"], unsafe_allow_html=True)


if __name__ == "__main__":
    authenticator.login()
    if st.session_state["authentication_status"]:
        authenticator.logout()
        st.write(f'Bienvenue *{st.session_state["name"]}*')
        main()


