import docker
import json
import os


import argparse

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def transcribe_docker(audio_file_name):
    logging.info("Starting transcription process")
    client = docker.from_env()
    
    current_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    logging.info(f"Current directory: {current_dir}")

    logging.info(f"Running Docker container for file: {audio_file_name}")
    transcribe_path = os.path.join('containerised_steps', 'transcribe', 'transcribe.py')
    logging.info(f"Transcribe path: {transcribe_path}")
    container = client.containers.run(
        'whisperx-transcriber',
        command=["python",transcribe_path,  audio_file_name],# Liste récursivement le contenu de /app
        volumes={
            current_dir: {'bind': '/app', 'mode': 'ro'},
            os.path.join(current_dir, 'data'): {'bind': '/data', 'mode': 'ro'}
        },
        remove=True,
        stdout=True,
        stderr=True
    )

    logging.info("Container execution completed")
    output = container.decode('utf-8')
    try:
        logging.info("Attempting to parse JSON output")
        logging.info(output)
        logging.info("JSON parsing successful")
        return output
    except json.JSONDecodeError:
        logging.error("Failed to decode JSON output")
        return {"Error": "JSON not decoded"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="transcription dans docker avec whisperx")
    parser.add_argument("--name", type=str, default="DCR_POC_CRA_1.wav")
    
    args = parser.parse_args()
    
    resultat = transcribe_docker(args.name)
    print(json.dumps(resultat, indent=2, ensure_ascii=False))