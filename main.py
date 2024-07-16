import streamlit as st
import pathlib
import streamlit_authenticator as stauth
import asyncio
from pydub import AudioSegment

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
# from workflows.workflowsLLM import (llm_pipe,
#                                     transcribe,
#                                     parse_transcription, 
#                                     formatted_verbatim,
#                                     c_transcription_list, 
#                                     parsed_cri,
#                                     parsed_cra,
#                                     parsed_cred)
import spacy
import epitran
from similarity.jarowinkler import JaroWinkler

from workflows.pipeline import Pipeline, Step,Parameters
from workflows.nlp_steps import (transcribe_empty,
                       parse_whisperx_output, 
                       format_for_output, 
                       spell_correct, 
                       step_llm_inference)
import logging
from conf import (cfg, 
                senators_file_path)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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

    llm_pipe >>  transcribe >> parse_transcription >> formatted_verbatim# >> c_transcription_list 
    llm_pipe | parse_transcription >> parsed_cri + parsed_cra + parsed_cred
    pipe = llm_pipe

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

        try:

            pipeline_future = asyncio.create_task(pipe.start("data/transcribe_encoded.json"))
            
            transcribe_output = await transcribe.output
            logger.info(f"step1 output: ======> {len(transcribe_output)}")
            logger.info(f"step2 output: ======> {len(await parse_transcription.output)}")
            #logger.info(f"step3 output: ======> {len(await formatted_verbatim.output)}")
            #logger.info(f"step4 output: ======> {len(await c_transcription_list.output)}")

            st.session_state["verbatim"] =  await formatted_verbatim.output
            logger.info(f"async value streamlit: ======> {str(formatted_verbatim)}")
            st.session_state["c_verbatim"] = await c_verbatim_output.output
            logger.info(f"async value streamlit: ======> {str(c_verbatim_output)}")
            st.session_state["c_cri"] = await parsed_cri.output
            logger.info(f"async value streamlit: ======> {str(parsed_cri)}")
            st.session_state["c_cra"] = await parsed_cra.output
            logger.info(f"async value streamlit: ======> {str(parsed_cra)}")
            st.session_state["c_cred"] = await parsed_cred.output    
            logger.info(f"async value streamlit: ======> {str(parsed_cred)}")

            result = await pipeline_future

        except asyncio.CancelledError as e:
            logger.info(f"Pipeline execution was cancelled {e}")
        except Exception as e:
            logger.error(f"An error occurred: {e}")
        finally:
    
            pipe.cancel_steps()
            pipeline_future.cancel()
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


