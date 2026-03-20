#!/usr/bin/env python3
"""
Voice chatbot — Phase 2: Adaptive Listening via VAD

Changes vs Phase 1 (chatbot-moderate-qwen.py):
  - record_audio() replaced with record_audio_vad() using silero-vad.
  - Recording stops dynamically when the user stops speaking (~600 ms silence).
  - stt() updated to use whisper.transcribe() for variable-length audio.
  - Per-turn timing log added for thesis runtime analysis.

Hardware target: Nvidia RTX 3060 (6 GB VRAM), Windows 11 / Linux.

Install (run once inside edge_env):
    pip install silero-vad
    pip install git+https://github.com/openai/whisper.git
    pip install TTS transformers torch sounddevice numpy accelerate

Run:
    python chatbot-vad-qwen.py

Versions tested:
    Python       3.12.9
    Torch        2.8.0
    torchaudio   2.8.0
    numpy        2.2.5
    coqui-tts2   0.27.2
    silero-vad   5.x
    whisper      release 20250625
    transformers 4.55.5
"""

import time
import torch
import torchaudio
import sounddevice as sd
import numpy as np
from TTS.api import TTS
from typing import List
import whisper
from transformers import AutoModelForCausalLM, AutoTokenizer
from silero_vad import load_silero_vad, get_speech_timestamps

print(f"Torch: {torch.__version__}  |  torchaudio: {torchaudio.__version__}")

# ─────────────────────  Audio Config ───────────────────────────

STT_MODEL        = "base"       # Whisper size: tiny/base/small/medium/large
TTS_MODEL        = "tts_models/multilingual/multi-dataset/your_tts"
SAMPLE_RATE      = 16000        # Hz — required by silero-vad and Whisper
CHANNELS         = 1

# ───────────────────── VAD Parameters ───────────────────────────
# Silero-vad requires chunk sizes of exactly 256 or 512 samples at 16 kHz.
CHUNK_SIZE       = 512          # ~32 ms per chunk at 16 kHz
VAD_THRESHOLD    = 0.5          # speech probability threshold (0–1)
# How many consecutive silent chunks before we stop recording.
# 20 chunks × 32 ms = ~640 ms of silence → feels natural in conversation.
SILENCE_CHUNKS   = 20
# Safety net: never record longer than this even if VAD keeps firing.
MAX_RECORD_SECS  = 15

# ─────────────────────  LM Config ───────────────────────────

LM_MODEL   = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_HISTORY = 6
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

GEN_KWARGS = {
    "repetition_penalty": 1.05,
    "do_sample":          True,
    "temperature":        0.3,
    "max_new_tokens":     45,
    "top_p":              0.7,
}

# ─────────────────────  Load Models ───────────────────────────

print("Loading silero-vad...")
vad_model = load_silero_vad()            # lightweight ~1.5 MB, runs on CPU

print("Loading Whisper STT...")
stt_model = whisper.load_model(STT_MODEL, device=DEVICE)

print("Loading TTS model...")
tts = TTS(TTS_MODEL).to(DEVICE)
print("TTS speakers:", tts.speakers)

# ─────────────────────  LM Loading ───────────────────────────

def load_lm_model(model_name: str = LM_MODEL, device: str = DEVICE):
    """Load Qwen2.5-Instruct with fp16 on GPU via device_map='auto'."""
    print(f"Loading tokenizer & model '{model_name}' ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    return tokenizer, model


# ─────────────────────  VAD-Driven Recording ───────────────────────────

def record_audio_vad() -> np.ndarray:
    """
    Stream microphone input and stop recording automatically when the
    user has been silent for SILENCE_CHUNKS consecutive chunks.

    Returns a float32 numpy array at SAMPLE_RATE Hz suitable for Whisper.
    """
    print("Listening... (speak now, recording will stop when you're done)")

    recorded_chunks: List[np.ndarray] = []
    silent_count    = 0
    speech_detected = False
    max_chunks      = int(MAX_RECORD_SECS * SAMPLE_RATE / CHUNK_SIZE)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype="float32", blocksize=CHUNK_SIZE) as stream:
        for _ in range(max_chunks):
            chunk, _ = stream.read(CHUNK_SIZE)
            chunk_1d = chunk.squeeze()            # shape (CHUNK_SIZE,)

            # silero-vad expects a 1-D float32 tensor
            chunk_tensor = torch.from_numpy(chunk_1d)
            with torch.no_grad():
                speech_prob = vad_model(chunk_tensor, SAMPLE_RATE).item()

            is_speech = speech_prob >= VAD_THRESHOLD

            if is_speech:
                speech_detected = True
                silent_count = 0
                recorded_chunks.append(chunk_1d)
            else:
                if speech_detected:
                    # Only count silence after we've heard at least one speech chunk
                    silent_count += 1
                    recorded_chunks.append(chunk_1d)  # include trailing silence
                    if silent_count >= SILENCE_CHUNKS:
                        break
                # Before any speech starts, keep listening (don't accumulate silence)

    if not recorded_chunks:
        # No speech heard at all within MAX_RECORD_SECS → return short silence
        print("[VAD] No speech detected within timeout.")
        return np.zeros(CHUNK_SIZE, dtype=np.float32)

    audio = np.concatenate(recorded_chunks, axis=0)
    elapsed = len(audio) / SAMPLE_RATE
    print(f"[VAD] Recorded {elapsed:.2f}s of audio (including trailing silence buffer).")
    return audio


# ─────────────────────  STT ───────────────────────────

def stt(audio_np: np.ndarray) -> str:
    """
    Transcribe variable-length audio using whisper.transcribe().
    We use transcribe() (not the low-level decode() used in Phase 1)
    because it handles arbitrary audio lengths without pad_or_trim artefacts.
    """
    print("Transcribing with Whisper...")
    result = stt_model.transcribe(
        audio_np,
        language="en",
        fp16=(DEVICE == "cuda"),
    )
    text = result["text"].strip()
    print(f"You said: {text}")
    return text


# ─────────────────────  TTS ───────────────────────────

def speak(text: str) -> None:
    print(f"Bot: {text}")
    wav = tts.tts(text, language="en", speaker="male-en-2")
    sd.play(np.array(wav), samplerate=tts.synthesizer.output_sample_rate)
    sd.wait()


# ─────────────────────  Prompt Engineering ───────────────────────────

def build_chat_input(tokenizer, history: List[str], user_input: str):
    """
    Build the tokenizer input from alternating [user, bot, user, bot, …] history.
    Prefers tokenizer.apply_chat_template (Qwen's native helper).
    """
    messages = []
    for i, msg in enumerate(history):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": msg})
    messages.append({"role": "user", "content": user_input})

    try:
        if hasattr(tokenizer, "apply_chat_template"):
            chat_prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
            if isinstance(chat_prompt, dict):
                return chat_prompt.get("text") or str(chat_prompt)
            return chat_prompt
    except Exception as e:
        print(f"[Chat template] Falling back to manual format: {e}")

    system = "You are a helpful assistant. Answer conversationally and concisely."
    parts = [f"SYSTEM: {system}", ""]
    for m in messages:
        parts.append(f"{m['role'].upper()}: {m['content'].strip()}")
    parts.append("ASSISTANT:")
    return "\n".join(parts)


# ─────────────────────  LM Response ───────────────────────────

def generate_response(user_input: str, chat_history: List[str],
                      lm, tokenizer) -> str:
    prompt_text = build_chat_input(tokenizer, chat_history, user_input)

    if not isinstance(prompt_text, str):
        try:
            prompt_text = tokenizer.decode(prompt_text, skip_special_tokens=False)
        except Exception:
            prompt_text = str(prompt_text)

    inputs = tokenizer(prompt_text, return_tensors="pt")
    try:
        model_device = next(lm.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
    except StopIteration:
        pass

    with torch.no_grad():
        output = lm.generate(**inputs, **GEN_KWARGS)

    input_len = inputs["input_ids"].shape[1]
    try:
        response = tokenizer.decode(output[0][input_len:],
                                    skip_special_tokens=True).strip()
    except Exception:
        full = tokenizer.decode(output[0], skip_special_tokens=True)
        response = full
        if isinstance(full, str) and full.startswith(prompt_text.strip()):
            response = full[len(prompt_text.strip()):].strip()

    chat_history.append(user_input)
    chat_history.append(response)
    if len(chat_history) > MAX_HISTORY:
        chat_history = chat_history[-MAX_HISTORY:]

    return response


# ─────────────────────  Main Loop ───────────────────────────

def main():
    print("Loading language model...")
    tokenizer, lm_model = load_lm_model()

    chat_history: List[str] = []
    print("\nVoice chatbot ready (Phase 2 — VAD Adaptive Listening)")
    print("Say 'quit', 'exit', or 'stop' to end the session.\n")

    while True:
        # ──────────────────────── Record ────────────────────────
        t0 = time.perf_counter()
        audio = record_audio_vad()
        t_record = time.perf_counter() - t0

        # ──────────────────────── STT ───────────────────────────
        t1 = time.perf_counter()
        text = stt(audio)
        t_stt = time.perf_counter() - t1

        if not text:
            print("[Loop] Empty transcription — listening again.")
            continue

        # Quit commands
        if text.lower().rstrip(".,!?") in ("quit", "exit", "stop"):
            speak("Goodbye!")
            break

        # ──────────────────────── LM ───────────────────────────
        t2 = time.perf_counter()
        reply = generate_response(text, chat_history, lm_model, tokenizer)
        t_lm = time.perf_counter() - t2

        # ──────────────────────── TTS ───────────────────────────
        t3 = time.perf_counter()
        speak(reply)
        t_tts = time.perf_counter() - t3

        # ──────── Timing log (for thesis runtime analysis) ────────
        print(
            f"[Timing] Record: {t_record:.2f}s | "
            f"STT: {t_stt:.2f}s | "
            f"LM: {t_lm:.2f}s | "
            f"TTS: {t_tts:.2f}s | "
            f"Total: {t_record + t_stt + t_lm + t_tts:.2f}s"
        )


if __name__ == "__main__":
    main()
