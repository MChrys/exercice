import sys
import json
import whisperx
import torch
import os
import gc
from main import transcribe

def transcribe(audio_file_path, model_name, device, language, compute_type, batch_size):
    model = whisperx.load_model(model_name, device, language=language, compute_type=compute_type)
    audio = whisperx.load_audio(audio_file_path)
    result = model.transcribe(audio, batch_size=batch_size)
    gc.collect()
    torch.cuda.empty_cache()
    del model
    model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    gc.collect()
    torch.cuda.empty_cache()
    del model_a
    diarize_model = whisperx.DiarizationPipeline( device=device)
    diarize_segments = diarize_model(audio)
    gc.collect()
    torch.cuda.empty_cache()
    del diarize_model
    result = whisperx.assign_word_speakers(diarize_segments, result)
    return result

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python transcribe.py <audio_file_path>")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    if not os.path.exists(audio_path):
        print(f"Error: File {audio_path} not found")
        sys.exit(1)
    
    transcribe(audio_path)