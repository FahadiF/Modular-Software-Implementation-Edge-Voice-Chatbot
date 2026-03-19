#!/usr/bin/env python3
"""
Voice chatbot with:
1. **Local Whisper STT** (openai/whisper)
2. **High‑quality TTS** using **Coqui TTS** (tts library)

Versions:
NVidia driver 535.274.02 CUDA Version: 12.2
numpy                      2.2.5
nvidia-cublas-cu12         12.8.4.1
nvidia-cuda-cupti-cu12     12.8.90
nvidia-cuda-nvrtc-cu12     12.8.93
nvidia-cuda-runtime-cu12   12.8.90
nvidia-cudnn-cu12          9.10.2.21
nvidia-cufft-cu12          11.3.3.83
nvidia-cufile-cu12         1.13.1.3
nvidia-curand-cu12         10.3.9.90
nvidia-cusolver-cu12       11.7.3.90
nvidia-cusparse-cu12       12.5.8.93
nvidia-cusparselt-cu12     0.7.1
nvidia-nccl-cu12           2.27.3
nvidia-nvjitlink-cu12      12.8.93
nvidia-nvtx-cu12           12.8.90
Python 3.12.9
Torch 2.8.0
torchvision 0.23.0
torchaudio 2.8.0
coqui-tts2 0.27.2
whisper release 20250625
transformers 4.55.5

Install:
    pip install git+https://github.com/openai/whisper.git
    pip install TTS transformers torch sounddevice numpy

If using GPU:
    pip install torch --index-url https://download.pytorch.org/whl/cu121

Run:
    python voice_chatbot.py
"""

import torch
import torchaudio
import queue
import sounddevice as sd
import numpy as np
from TTS.api import TTS
from typing import List
import whisper
from transformers import AutoModelForCausalLM, AutoTokenizer

print(torch.version.__version__)
print(torchaudio.version.__version__)

# ---------------------- Audio Config ----------------------------
STT_MODEL = "base"          # Whisper model size
WHISPER_MODEL = "base"  # 'tiny', 'base', 'small', 'medium', 'large'
TTS_MODEL = "tts_models/multilingual/multi-dataset/your_tts"  # Coqui high‑quality
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_DURATION = 5.0      # seconds per listen. Increase for longer user utterances.

# ------------------------- LM Config ----------------------------
#LM_MODEL = "microsoft/DialoGPT-medium"
#LM_MODEL = "Qwen/Qwen2.5-3B-Instruct-AWQ" #for Linux Only
LM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

MAX_HISTORY = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Generation parameters (tweak to taste)
GEN_KWARGS = {
    #"max_new_tokens": 256,
    #"temperature": 0.7,
    #"top_p": 0.9,
    "repetition_penalty": 1.05,
    "do_sample": True,
    #"messages": [
    #    {"role": "system", "content": "You are concise. Answer like a natural human in short replies."},
        #{"role": "user", "content": user_prompt}
    #],
    "temperature": 0.3,
    "max_new_tokens": 45,
    "top_p": 0.7,
}

# ---------------------------- Load models ----------------------------
print("Loading Whisper STT...")
stt_model = whisper.load_model(STT_MODEL, device=DEVICE)

print("Loading TTS model...")
tts = TTS(TTS_MODEL).to(DEVICE)
print(tts.speakers)

# ---------------------------- Language Model Init ---------------------
def load_lm_model(model_name: str = LM_MODEL, device: str = DEVICE):
    """
    Load Qwen2.5 AWQ with trust_remote_code=True and device_map='auto'.
    Assumes model provides HF-compatible interfaces.
    """
    print(f"Loading tokenizer & model '{model_name}' (trust_remote_code=True) ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Use device_map='auto' so HF places tensors on GPU/CPU appropriately.
    # Use fp16 to reduce memory.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    # If model was loaded with device_map='auto' it may already be on the proper device(s).
    return tokenizer, model


# ---------------------------- Audio Capture ----------------------------

def record_audio(duration=5):
    print("Listening...")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    return audio.squeeze()


def stt(audio_np):
    print("Transcribing locally with Whisper...")
    audio = whisper.pad_or_trim(audio_np)
    mel = whisper.log_mel_spectrogram(audio).to(DEVICE)
    result = stt_model.decode(mel)
    text = result.text.strip()
    print("You said:", text)
    return text

# ---------------------------- TTS ----------------------------

def speak(text):
    print(f"Bot: {text}")
    wav = tts.tts(text, language="en", speaker="male-en-2")
    sd.play(np.array(wav), samplerate=tts.synthesizer.output_sample_rate)
    sd.wait()

# --------------------- prompt engineering --------------------
def build_chat_input(tokenizer, history: List[str], user_input: str):
    """
    Prefer tokenizer.apply_chat_template if available (Qwen repo helper).
    Otherwise fallback to a simple prompt template with user/assistant markers.
    `history` is a flat list like [u1, b1, u2, b2, ...] (alternating user/bot).
    """
    # Build messages sequence from history + the new user input
    messages = []
    for i, msg in enumerate(history):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": msg})
    # append new user message as well
    messages.append({"role": "user", "content": user_input})

    # Preferred: tokenizer has a chat template helper
    try:
        # Some Qwen tokenizers expose apply_chat_template
        if hasattr(tokenizer, "apply_chat_template"):
            # apply_chat_template may accept different args across versions; this is a best-effort call
            chat_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            # If apply_chat_template returns tokens or a dict, convert to string
            if isinstance(chat_prompt, dict):
                # many implementations return {"input_ids": ..., "text": "..."}
                chat_text = chat_prompt.get("text") or chat_prompt.get("input_ids") or str(chat_prompt)
            else:
                chat_text = chat_prompt
            return chat_text
    except Exception as e:
        # If anything fails, we'll fallback to manual construction below.
        print("Tokenizer chat-template helper not available or failed; using fallback prompt. (", e, ")")

    # Fallback: simple manual formatting
    # Use system instructions to help Qwen behave like a chatbot
    system = "You are a helpful assistant. Answer conversationally and concisely."
    parts = [f"SYSTEM: {system}", ""]
    for m in messages:
        role = m["role"].upper()
        content = m["content"].strip()
        parts.append(f"{role}: {content}")
    parts.append("ASSISTANT:")  # the generation will continue after this marker
    chat_text = "\n".join(parts)
    return chat_text

# ---------------------------- LM Response ----------------------------
def generate_response(user_input, chat_history, lm, tokenizer):

    prompt_text = build_chat_input(tokenizer, chat_history, user_input)

    if not isinstance(prompt_text, str):
      try:
        prompt_text = tokenizer.decode(prompt_text, skip_special_tokens=False)
      except Exception:
        prompt_text = str(prompt_text)

    # Tokenize
    # Some tokenizers return dict with input_ids, others require text.
    inputs = tokenizer(prompt_text, return_tensors="pt")
    # Move to appropriate device if necessary (model may already be on device_map)
    # If model uses device_map='auto', some model parts are on GPU, and inputs must be on that GPU.
    # To be safe, move inputs to the same device as the model's first param
    try:
        first_param = next(lm.parameters())
        model_device = first_param.device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
    except StopIteration:
        # model has no parameters? unlikely; keep inputs on CPU
        pass

    # Generate
    with torch.no_grad():
        output = lm.generate(**inputs, **GEN_KWARGS)

    # Decode: skip the prompt tokens in the generated output.
    # Many vendor tokenizers expect you to slice by input length:
    input_len = inputs["input_ids"].shape[1]
    try:
        out_ids = output[0][input_len:]
        response = tokenizer.decode(out_ids, skip_special_tokens=True).strip()
    except Exception:
        # fallback: decode whole generation and remove the prompt text if it appears
        full = tokenizer.decode(output[0], skip_special_tokens=True)
        response = full
        # try to remove prompt_text from the beginning if present
        if isinstance(full, str) and prompt_text.strip() and full.startswith(prompt_text.strip()):
            response = full[len(prompt_text.strip()):].strip()

    # Append to history
    chat_history.append(user_input)
    chat_history.append(response)
    # Truncate history to last MAX_HISTORY turns (pairs)
    if len(chat_history) > MAX_HISTORY:
        chat_history = chat_history[-MAX_HISTORY:]

    return response

# ---------------------------- Main Loop ----------------------------

def main():
    print("Loading language model...")
    tokenizer, lm_model = load_lm_model()

    # Keep chat history as alternating list [user, bot, user, bot, ...]
    chat_history: List[str] = []

    print("Voice chatbot ready — say 'quit' to exit.")
    while True:
        audio = record_audio(duration=SAMPLE_DURATION)
        text = stt(audio)
        if not text:
            continue
        if text.lower() in ["quit.", "exit.", "stop.", "quit!", "exit!", "stop!"]:
            speak("Goodbye!")
            break

        reply = generate_response(text, chat_history, lm_model, tokenizer)
        speak(reply)


if __name__ == "__main__":
    main()
