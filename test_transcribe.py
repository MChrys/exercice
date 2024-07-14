import whisperx

import argparse
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def transcribe(audio_file, model_name, device, compute_type):
    logging.info(f"Début de la transcription pour le fichier : {audio_file}")
    logging.info(f"Utilisation du modèle : {model_name}, dispositif : {device}, type de calcul : {compute_type}")

    # Chargement du modèle préalablement téléchargé
    logging.info("Chargement du modèle de transcription")
    model = whisperx.load_model(
            model_name, 
            device=device, 
            compute_type=compute_type, 
            download_root="./whisperx_models",
            asr_options=dict(
                max_new_tokens=128,
                clip_timestamps="0",
                hallucination_silence_threshold=0.2,
                hotwords=[]
            )
    )
    logging.info("Modèle de transcription chargé avec succès")
    
    logging.info("Début de la transcription")
    result = model.transcribe(audio_file, print_progress=True, num_workers=4)
    logging.info("Transcription terminée")

    # Chargement du modèle d'alignement préalablement téléchargé
    logging.info(f"Chargement du modèle d'alignement pour la langue : {result['language']}")
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device, model_dir="./whisperx_models")
    logging.info("Alignement des segments")
    result = whisperx.align(result["segments"], model_a, metadata, audio_file, device)
    logging.info("Alignement terminé")

    # Diarisation
    logging.info("Début de la diarisation")
    diarize_model = whisperx.DiarizationPipeline(model_name="pyannote/speaker-diarization", use_auth_token=False, device=device)
    diarize_segments = diarize_model(audio_file)
    logging.info("Diarisation terminée")

    # Assignation des locuteurs
    logging.info("Assignation des locuteurs aux segments")
    result = whisperx.assign_word_speakers(diarize_segments, result)
    logging.info("Assignation des locuteurs terminée")

    # Affichage des résultats
    logging.info("Affichage des résultats de la transcription")
    for segment in result["segments"]:
        logging.info(f"Locuteur {segment['speaker']}: {segment['text']}")

    logging.info("Transcription complète terminée")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download  whisperx model")
    parser.add_argument("--file", type=str, default="data/DCR_POC_CRA_1.wav")
    parser.add_argument("--model", type=str, default="large-v2")

    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--compute_type", type=str, default="int8", choices=["int8", "float32"])
    
    args = parser.parse_args()
    
    transcribe(args.file,args.model, args.device, args.compute_type)