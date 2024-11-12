import whisper
import torch

# Check if CUDA is available and set device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the Whisper Medium model and move it to the specified device
model = whisper.load_model("medium").to(device)

def interpret_audio(audio_file_path):
    # Perform transcription on the specified device
    result = model.transcribe(audio_file_path, fp16=torch.cuda.is_available())
    
    print("Transcription:", result['text'])
    return result

# Example usage
audio_file_path = "/home/matrix/Desktop/AI Pocket Tutor/AI-Pocket-Tutor/sample_audio/OSR_us_000_0010_8k.wav"
interpret_audio(audio_file_path)
