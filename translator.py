import os
os.environ["GPIOZERO_PIN_FACTORY"] = "lgpio"
import subprocess
import tempfile
import time
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from gpiozero import Button
from llama_cpp import Llama

# --- CONFIG ---
SAMPLE_RATE = 44100
CHUNK_SECONDS = 0.5
BUTTON_PIN = 23
GGUF_MODEL = "/home/edgemd/.cache/huggingface/hub/models--bartowski--Llama-3.2-3B-Instruct-GGUF/snapshots/5ab33fa94d1d04e903623ae72c95d1696f09f9e8/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
PIPER_BIN = "/home/edgemd/piper/piper/piper"
PIPER_MODEL = "/home/edgemd/piper/models/es_MX-claude-high.onnx"
WHISPER_BIN = "/home/edgemd/whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL = "/home/edgemd/whisper.cpp/models/ggml-base.en.bin"

button = Button(BUTTON_PIN)

print("Loading LLM from cache...")
llm = Llama(model_path=GGUF_MODEL, n_ctx=512, n_threads=4, verbose=False)
print("LLM loaded. Hold button to record.")

def record_while_held():
    frames = []
    chunk_size = int(CHUNK_SECONDS * SAMPLE_RATE)
    print("Recording...")
    while button.is_pressed:
        chunk = sd.rec(chunk_size, samplerate=SAMPLE_RATE, channels=1, dtype='int16')
        sd.wait()
        frames.append(chunk)
    print("Done recording.")
    return np.concatenate(frames)

def save_audio(audio):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav.write(tmp.name, SAMPLE_RATE, audio)
    return tmp.name

def transcribe(audio_path):
    result = subprocess.run(
        [WHISPER_BIN, "-m", WHISPER_MODEL, "-f", audio_path, "-l", "en", "-np", "-nt"],
        capture_output=True, text=True, check=True
    )
    return result.stdout.strip()

def translate_with_llama(english_text):
    prompt = (
        "You are a medical interpreter. Translate the following English text "
        "to Spanish. Output only the Spanish translation directly, nothing else no extra notes, speeches, rambles, or explanations.\n\n"
        f"English: {english_text}\nSpanish:"
    )
    response = llm(prompt, max_tokens=200, stop=["\n", "English:"], echo=False)
    return response["choices"][0]["text"].strip()

def speak_spanish(spanish_text):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out_wav = tmp.name
    subprocess.run(
        [PIPER_BIN, "--model", PIPER_MODEL, "--output_file", out_wav],
        input=spanish_text.encode(),
        check=True
    )
    subprocess.run(["aplay", "-D", "plughw:2,0", out_wav], check=True)
    os.unlink(out_wav)

if __name__ == "__main__":
    while True:
        try:
            button.wait_for_press()

            audio = record_while_held()
            if len(audio) == 0:
                continue

            start = time.time()
            audio_file = save_audio(audio)
            t1 = time.time()

            english = transcribe(audio_file)
            os.unlink(audio_file)
            t2 = time.time()
            print(f"You said: {english}")
            print(f"  [transcribe: {t2 - t1:.2f}s]")

            spanish = translate_with_llama(english)
            t3 = time.time()
            print(f"Spanish: {spanish}")
            print(f"  [translate: {t3 - t2:.2f}s]")

            speak_spanish(spanish)
            t4 = time.time()
            print(f"  [tts+play: {t4 - t3:.2f}s]")
            print(f"Total time: {t4 - start:.2f}s")

        except KeyboardInterrupt:
            print("\nExiting.")
            break
