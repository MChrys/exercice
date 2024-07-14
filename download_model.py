import whisperx
import argparse

def download(model_name, language_code, device, compute_type):

    print(f"DL transcription : {model_name}")
    whisperx.load_model(
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
    
    print(f"DL align : {language_code}")
    whisperx.load_align_model(language_code=language_code, device=device, model_dir="./whisperx_models")
    
    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download  whisperx model")
    parser.add_argument("--model", type=str, default="large-v2")
    parser.add_argument("--language", type=str, default="fr")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--compute_type", type=str, default="int8", choices=["int8", "float32"])
    
    args = parser.parse_args()
    
    download(args.model, args.language, args.device, args.compute_type)
    