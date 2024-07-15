import streamlit as st
import pathlib
import spacy
import epitran
import streamlit_authenticator as stauth


from pydub import AudioSegment
import os
import asyncio
from config import cfg
from workflows.workflowsLLM import (llm_workflow,
                                   formatted_verbatim,
                                   c_verbatim_output,
                                   parsed_cri,
                                   parsed_cra,
                                   parsed_cred)




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


            result = asyncio.create_task(llm_workflow.start(audio_path))
            st.session_state["verbatim"] =  await formatted_verbatim.output
            st.session_state["c_verbatim"] = await c_verbatim_output.output
            st.session_state["c_cri"] = await parsed_cri.output
            st.session_state["c_cra"] = await parsed_cra.output
            st.session_state["c_cred"] = await parsed_cred.output
            await result

 
    
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


