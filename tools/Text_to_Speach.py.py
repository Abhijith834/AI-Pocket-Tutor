# Text_to_Speach.py
import sys
from TTS.api import TTS

if len(sys.argv) < 2:
    print("No text provided!")
    sys.exit(1)

text_content = sys.argv[1]  # The text content passed by the chat system

tts = TTS("tts_models/en/ljspeech/glow-tts")
print("Available speakers:", tts.speakers)
tts.tts_to_file(text_content, file_path="AI-Pocket-Tutor/tools/output.wav")
