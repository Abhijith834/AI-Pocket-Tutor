# Text_to_Speach.py
import sys
from TTS.api import TTS

# Require at least one argument: the text to speak
if len(sys.argv) < 2:
    sys.exit(1)

raw = sys.argv[1]

# Strip leading bracketed section if present
if raw.startswith('['):
    end = raw.find(']')
    raw = raw[end+1:].strip() if end != -1 else raw[1:].strip()

# Nothing to say?
if not raw:
    sys.exit(1)

# Generate speech
tts = TTS("tts_models/en/ljspeech/glow-tts")
tts.tts_to_file(raw, file_path="AI-Pocket-Tutor/tools/output.wav")
