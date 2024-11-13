import os
import whisper
from pathlib import Path

def process_audio(audio_path):
    """
    Transcribes the audio file to text and saves the output in a .txt file in the output folder.
    """
    model = whisper.load_model("base")  # Load a pre-trained Whisper model
    audio_path = str(audio_path)  # Convert Path object to string if needed

    # Transcribe the audio
    result = model.transcribe(audio_path)
    transcription = result['text']

    # Save the transcription to a text file
    output_dir = Path("output") / "audio_transcriptions"
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

    output_file = output_dir / f"{Path(audio_path).stem}_transcription.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(transcription)

    print(f"Transcription for {audio_path} saved to {output_file}")
