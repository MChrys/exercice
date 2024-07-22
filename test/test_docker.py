import docker
import json
import os


import argparse

import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from workflows.nlp_steps import transcribe_docker


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="transcription dans docker avec whisperx")
    parser.add_argument("--name", type=str, default="transcribe_encoded.json")
    
    args = parser.parse_args()
    
    resultat = transcribe_docker(args.name)
    logger.info(json.dumps(resultat, indent=2, ensure_ascii=False))