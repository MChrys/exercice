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
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
import os

from nlp_fonctions import transcribe, parse_whisperx_output, format_for_output, spell_correct, apply_parse_and_reformat

initialize(config_path="config")
cfg = compose(config_name="local")
api_key = cfg["api_key"]
device = cfg["device"]
batch_size = cfg["batch_size"]
compute_type = cfg["compute_type"]
model_name = cfg["model_name"]
senators_file_path = cfg["senators_file_path"]
llm_model_name = cfg["llm_model_name"]
hf_model_name = cfg["hf_model_name"]
max_tokens = cfg["max_tokens"]
max_concurrent_requests = cfg["max_concurrent_requests"]
language = cfg["language"]
jarowinkler = JaroWinkler()
epi = epitran.Epitran(cfg["epi"])
nlp = spacy.load(cfg["nlp"])


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
            w_result = transcribe(audio_path, 
                                  model_name,
                                  device,
                                  language,
                                  compute_type,
                                  batch_size
                                  )
            transcription_list = parse_whisperx_output(w_result)

            # Appliquer parse_text_between_cr_tags et reformater
            formatted_verbatim = format_for_output(transcription_list,
                                                   
                                                   )
            st.session_state["verbatim"] = formatted_verbatim

            c_transcription_list = spell_correct(transcription_list,
                                                  senators_file_path, 
                                                  epi, 
                                                  nlp, 
                                                  jarowinkler, 
                                                  verbose=True)

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

            cri_llm_inferance = ParallelLLMInference(llm_model_name, 
                                                     api_key, 
                                                     hf_model_name, 
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


