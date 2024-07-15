
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
from workflows.LLM_inf import ParallelLLMInference
from test_conf import compose, initialize
from omegaconf import DictConfig, OmegaConf
import os

from workflows.nlp_steps import transcribe, parse_whisperx_output, format_for_output, spell_correct, apply_parse_and_reformat

initialize(config_path="config")
cfg = compose(config_name="local")
print(cfg)