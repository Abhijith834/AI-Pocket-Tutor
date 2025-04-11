import os
import sys
import torch
import whisper

def transcribe_audio_file(input_filepath: str,
                          output_txt_filepath: str,
                          model_name: str = "small",
                          language: str = "en"):
    """
    Transcribes an audio file and writes the transcription to a text file.

    Parameters:
      input_filepath (str): Path to the input audio file.
      output_txt_filepath (str): Path where the transcription text file will be saved.
      model_name (str): Whisper model to use (default "small").
      language (str): Language code to use for transcription (default "en").

    Returns:
      transcription (str): The transcribed text.
    """
    # Choose CUDA if available, otherwise CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load the Whisper model on the appropriate device
    print(f"Loading Whisper model '{model_name}'...")
    try:
        model = whisper.load_model(model_name, device=device)
    except Exception as e:
        print("Error loading Whisper model:", e)
        sys.exit(1)
    
    # Transcribe the audio file
    print(f"Transcribing audio file: {input_filepath}")
    result = model.transcribe(input_filepath, language=language, fp16=True if device=="cuda" else False)
    transcription = result["text"].strip()
    
    # Write the transcription to the output text file
    try:
        with open(output_txt_filepath, "w", encoding="utf-8") as f:
            f.write(transcription)
        print(f"Transcription saved to: {output_txt_filepath}")
    except Exception as e:
        print("Error writing transcription file:", e)
    
    return transcription

if __name__ == "__main__":
    # This enables running the module directly from the command line:
    # Usage: python audio_transcriber.py <input_audio_file> <output_text_file>
    if len(sys.argv) < 3:
        print("Usage: python audio_transcriber.py <input_audio_file> <output_text_file>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    transcribe_audio_file(input_file, output_file)
