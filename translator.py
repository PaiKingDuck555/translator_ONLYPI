import os
import subprocess
import tempfile
import time
import sounddevice as sd
import scipy.io.wavfile as wav
import whisper
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- CONFIG ---
SAMPLE_RATE = 16000
RECORD_SECONDS = 6
HF_MODEL_NAME = "models--bartowski--Llama-3.2-3B-Instruct-GGUF"  # update to match your cached model name exactly
PIPER_MODEL = "/home/pi/piper/es_MX-ald-medium.onnx"   # update to your Spanish .onnx path

# Load LLM once at startup so it's not reloaded every call
print("Loading LLM from cache...")
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME, local_files_only=True)
llm = AutoModelForCausalLM.from_pretrained(
    HF_MODEL_NAME,
    local_files_only=True,
    torch_dtype=torch.float32,  # use float16 if your RPi/device supports it
    device_map="cpu"
)
llm.eval()
print("LLM loaded.")

def record_audio():
    print("Recording... speak now.")
    audio = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    print("Done recording.")
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav.write(tmp.name, SAMPLE_RATE, audio)
    return tmp.name

def transcribe(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, language="en")
    return result["text"].strip()

def translate_with_llama(english_text):
    prompt = (
        "You are a medical interpreter. Translate the following English text "
        "to Spanish. Output only the Spanish translation directly, nothing else no extra notes, speeches, rambles, or explanations.\n\n"
        f"English: {english_text}\nSpanish:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    with torch.no_grad():
        output = llm.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    # Extract only the Spanish part after the prompt
    spanish = decoded.split("Spanish:")[-1].strip()
    return spanish

def speak_spanish(spanish_text):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out_wav = tmp.name
    subprocess.run(
        ["piper", "--model", PIPER_MODEL, "--output_file", out_wav],
        input=spanish_text.encode(),
        check=True
    )
    subprocess.run(["aplay", out_wav], check=True)
    os.unlink(out_wav)

if __name__ == "__main__":
    start = time.time()

    audio_file = record_audio()
    english = transcribe(audio_file)
    os.unlink(audio_file)
    print(f"You said: {english}")

    spanish = translate_with_llama(english)
    print(f"Spanish: {spanish}")

    speak_spanish(spanish)

    print(f"Total time: {time.time() - start:.2f}s")
