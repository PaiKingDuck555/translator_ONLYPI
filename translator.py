import os
import subprocess
import tempfile
import time
import threading
import tkinter as tk
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import gpiod
from gpiod.line import Direction, Value, Bias
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
AUDIO_OUT = "plughw:3,0"

request = gpiod.request_lines(
    "/dev/gpiochip0",
    consumer="translator",
    config={BUTTON_PIN: gpiod.LineSettings(direction=Direction.INPUT, bias=Bias.PULL_UP)}
)

print("Loading LLM from cache...")
llm = Llama(model_path=GGUF_MODEL, n_ctx=512, n_threads=4, verbose=False)
print("LLM loaded. Hold button to record.")

# --- UI SETUP ---
root = tk.Tk()
root.attributes("-fullscreen", True)
root.configure(bg="black")
root.bind("<Escape>", lambda e: root.destroy())

status_var = tk.StringVar(value="Hold button to speak...")
english_var = tk.StringVar(value="")
spanish_var = tk.StringVar(value="")

tk.Label(root, textvariable=status_var, font=("Helvetica", 18), fg="gray", bg="black").pack(pady=(20, 0))
tk.Frame(root, bg="black", height=10).pack()

top_frame = tk.Frame(root, bg="#111111", bd=2, relief="flat")
top_frame.pack(fill="both", expand=True, padx=20, pady=(10, 5))
tk.Label(top_frame, text="ENGLISH", font=("Helvetica", 14, "bold"), fg="#888888", bg="#111111").pack(anchor="w", padx=15, pady=(10, 0))
tk.Label(top_frame, textvariable=english_var, font=("Helvetica", 26), fg="white", bg="#111111", wraplength=740, justify="left").pack(anchor="w", padx=15, pady=(5, 15))

bottom_frame = tk.Frame(root, bg="#111122", bd=2, relief="flat")
bottom_frame.pack(fill="both", expand=True, padx=20, pady=(5, 20))
tk.Label(bottom_frame, text="ESPAÑOL", font=("Helvetica", 14, "bold"), fg="#8888ff", bg="#111122").pack(anchor="w", padx=15, pady=(10, 0))
tk.Label(bottom_frame, textvariable=spanish_var, font=("Helvetica", 26), fg="#aaaaff", bg="#111122", wraplength=740, justify="left").pack(anchor="w", padx=15, pady=(5, 15))

def update_display(english, spanish, status="Hold button to speak..."):
    english_var.set(english)
    spanish_var.set(spanish)
    status_var.set(status)

# --- PIPELINE ---
def is_pressed():
    return request.get_value(BUTTON_PIN) == Value.INACTIVE

def record_while_held():
    frames = []
    chunk_size = int(CHUNK_SECONDS * SAMPLE_RATE)
    print("Recording...")
    while is_pressed():
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
    subprocess.run(["aplay", "-D", AUDIO_OUT, out_wav], check=True)
    os.unlink(out_wav)

def pipeline_loop():
    while True:
        while not is_pressed():
            time.sleep(0.05)

        root.after(0, status_var.set, "Recording...")
        audio = record_while_held()
        if len(audio) == 0:
            continue

        root.after(0, status_var.set, "Transcribing...")
        start = time.time()
        audio_file = save_audio(audio)
        t1 = time.time()

        english = transcribe(audio_file)
        os.unlink(audio_file)
        t2 = time.time()
        print(f"You said: {english}")
        print(f"  [transcribe: {t2 - t1:.2f}s]")

        root.after(0, status_var.set, "Translating...")
        spanish = translate_with_llama(english)
        t3 = time.time()
        print(f"Spanish: {spanish}")
        print(f"  [translate: {t3 - t2:.2f}s]")

        root.after(0, update_display, english, spanish, "Speaking...")
        speak_spanish(spanish)
        t4 = time.time()
        print(f"  [tts+play: {t4 - t3:.2f}s]")
        print(f"Total time: {t4 - start:.2f}s")

        root.after(0, status_var.set, "Hold button to speak...")

if __name__ == "__main__":
    t = threading.Thread(target=pipeline_loop, daemon=True)
    t.start()
    try:
        root.mainloop()
    finally:
        request.release()
