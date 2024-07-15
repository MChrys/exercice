import sys

import os

import logging

# Logging configuration
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def transcribe(audio_name):
    audio_path = os.path.abspath(f"/data/{audio_name}")
    if not os.path.exists(audio_path):
        logging.error(f"file {audio_path} does not exist")
        return

    if not os.path.isfile(audio_path):
        logging.error(f"'{audio_path}' is not a file")


    valid_extensions = ['.wav', '.mp3']
    if not any(audio_path.lower().endswith(ext) for ext in valid_extensions):
        logging.error(f"'{audio_path}' extension not supported {', '.join(valid_extensions)}")
        
    
    logging.info(f"Starting transcription of file: {audio_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Transcribe audio file")
    parser.add_argument("audio_name", type=str)
    
    args = parser.parse_args()
    

    
    transcribe(args.audio_name)