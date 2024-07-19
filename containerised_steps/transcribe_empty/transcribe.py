import sys

import os
import json 
import subprocess

def print_directory_tree(startpath):
    """Affiche l'arborescence d'un répertoire en utilisant la commande tree"""
    try:
        result = subprocess.run(['tree', startpath], capture_output=True, text=True)
        print(f"Arborescence  {startpath}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error tree: {e}")

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def transcribe(audio_name):
    logger.info(f"Starting transcription of file: {audio_name}")
    audio_path = os.path.abspath(f"/data/{audio_name}")
    #print_directory_tree('/app')
    with open(audio_path, 'r') as audio_file:
        result = json.load(audio_file)
    # if not os.path.exists(audio_path):
    #     logging.error(f"file {audio_path} does not exist")
    #     return

    # if not os.path.isfile(audio_path):
    #     logging.error(f"'{audio_path}' is not a file")


    # valid_extensions = ['.wav', '.mp3']
    # if not any(audio_path.lower().endswith(ext) for ext in valid_extensions):
    #     logging.error(f"'{audio_path}' extension not supported {', '.join(valid_extensions)}")
    logger.info(f"Starting transcription of file: {audio_path}")

    output_path = '/output/result.json'
    print_directory_tree('/output')
    with open(output_path, 'w') as f:
        json.dump(result, f)
    logger.info(f"Transcription of file: {output_path} done")
    print_directory_tree('/output')


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Transcribe audio file")
    parser.add_argument("audio_name", type=str)
    
    args = parser.parse_args()
    

    
    transcribe(args.audio_name)