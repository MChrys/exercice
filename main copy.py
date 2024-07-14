import streamlit as st
import pathlib
import spacy
import epitran
import json
import streamlit_authenticator as stauth

from similarity.jarowinkler import JaroWinkler
from pydub import AudioSegment
from hydra import compose, initialize
import os
import asyncio
from pipeline import Pipeline, Step,Parameters
from nlp_steps import (transcribe, 
                       parse_whisperx_output, 
                       format_for_output, 
                       spell_correct, 
                       step_llm_inference)

#initialize(config_path="config")
cfg = compose(config_name="local")
api_key = cfg["llm_api"]["api_key"]
llm_model_name = cfg["llm_api"]["llm_model_name"]
base_url = cfg["llm_api"]["base_url"]
device = cfg["device"]
batch_size = cfg["batch_size"]
compute_type = cfg["compute_type"]
model_name = cfg["model_name"]
senators_file_path = cfg["senators_file_path"]

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

authenticator = stauth.Authenticate(
    cfg['credentials']['credentials'],
    cfg['credentials']['cookie']['name'],
    cfg['credentials']['cookie']['key'],
    cfg['credentials']['cookie']['expiry_days'],
    cfg['credentials']['preauthorized']
)
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



async def main():
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



            llm_workflow >> transcription >> parse_transcription >> formatted_verbatim >> c_transcription_list 
            llm_workflow | c_transcription_list >> c_verbatim_output + parsed_cri + parsed_cra + parsed_cred

            await llm_workflow.start(audio_path)
            st.session_state["verbatim"] =  await formatted_verbatim.output
            st.session_state["c_verbatim"] = await c_verbatim_output.output
            st.session_state["c_cri"] = await parsed_cri.output
            st.session_state["c_cra"] = await parsed_cra.output

 
    
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
        asyncio.run(main())


