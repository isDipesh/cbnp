import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=device)
text = "न्यायपालिकाको आँखाबाट देखिने सङ्क्रमणकालीन नेपाली राजनीतिकी साक्षी हुन्- सुशीला कार्की । न्यायिक समाज निर्माणका लागि सत्यको जगमा उभिंदा महाअभियोगसम्म सामना गर्न तयार भएकी उनी इमान, निष्ठा र न्यायकी प्रतिमूर्ति हुन् ।"

AUDIO_PROMPT_PATH = "1.wav"
wav = multilingual_model.generate(
    text, audio_prompt_path=AUDIO_PROMPT_PATH, language_id="hi"
)
ta.save("test-3.wav", wav, multilingual_model.sr)
