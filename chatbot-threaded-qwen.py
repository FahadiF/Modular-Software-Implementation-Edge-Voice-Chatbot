#!/usr/bin/env python3
"""
Voice chatbot — Phase 3: Concurrent Multithreaded Pipeline

Architecture: Four independent threads communicating via thread-safe Queues.

  Thread A (Capture)  →[audio_queue]→  Thread B (Inference)  →[tts_queue]→  Thread C (Output)
     VAD + mic                           Whisper + Qwen                        Coqui + playback
  Thread D (Keyboard)
     Space = stop speech  |  q = quit

Key improvements vs Phase 2:
  - Thread A resumes listening immediately after dropping audio into the queue.
    The user can start the next utterance while Thread B is still running Whisper.
  - Backpressure via maxsize=1 queues: upstream blocks rather than accumulating
    stale utterances when downstream is busy.
  - Echo suppression: mic capture is muted while the bot is speaking to prevent
    the VAD from picking up TTS output through the microphone.
  - Keyboard interrupt (Thread D): Space stops current speech; q exits cleanly.
    Uses Windows built-in msvcrt — no extra pip package needed.
  - Clean shutdown via threading.Event (quit_event) + cascading None sentinels.
  - Per-thread timing logs for thesis runtime analysis.

Hardware target: Nvidia RTX 3060 (6 GB VRAM), Windows 11 / Linux.
Steady-state VRAM:  Whisper base ~145 MB  |  Qwen 1.5B fp16 ~3.0 GB  |  Coqui ~150 MB
Total GPU:          ~3.3 GB — safely within the 6 GB budget.

Install (run once inside edge_env):
    pip install silero-vad
    pip install git+https://github.com/openai/whisper.git
    pip install TTS transformers torch sounddevice numpy accelerate

Run:
    python chatbot-threaded-qwen.py

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

import re
import msvcrt
import threading
import queue
import time
import torch
import torchaudio
import sounddevice as sd
import numpy as np
from TTS.api import TTS
from typing import List, Optional
import whisper
from transformers import AutoModelForCausalLM, AutoTokenizer
from silero_vad import load_silero_vad

print(f"Torch: {torch.__version__}  |  torchaudio: {torchaudio.__version__}")

# ─────────────────────  Audio Config ───────────────────────────

STT_MODEL   = "base"       # Whisper size: tiny/base/small/medium/large
TTS_MODEL   = "tts_models/multilingual/multi-dataset/your_tts"
SAMPLE_RATE = 16000        # Hz — required by silero-vad and Whisper
CHANNELS    = 1

# ─────────────────────  VAD Parameters ──────────────────────────
# Silero-vad requires chunk sizes of exactly 256 or 512 samples at 16 kHz.
CHUNK_SIZE    = 512         # ~32 ms per chunk at 16 kHz
VAD_THRESHOLD = 0.5         # speech probability threshold (0–1)
# How many consecutive silent chunks before we stop recording.
# 30 chunks × 32 ms = ~960 ms of silence — slightly longer than Phase 2
# to avoid cutting off natural mid-sentence pauses.
SILENCE_CHUNKS = 30
# Safety net: never record longer than this even if VAD keeps firing.
MAX_RECORD_SECS = 15
# Minimum VAD-confirmed speech chunks before forwarding to Whisper.
# 12 chunks × 32 ms = ~384 ms — filters out fan-noise / HVAC blips.
MIN_SPEECH_CHUNKS = 12

# ─────────────────────  LM Config ───────────────────────────

LM_MODEL    = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_HISTORY = 6
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

GEN_KWARGS = {
    "repetition_penalty": 1.05,
    "do_sample":          True,
    "temperature":        0.3,
    "max_new_tokens":     45,
    "top_p":              0.7,
}

# ─────────────────────  Queues & Shared Events ──────────────────
#
#  audio_queue  (Thread A → Thread B)
#    Payload : np.ndarray, float32, 1-D
#    maxsize=1: Thread A blocks if Thread B is still processing.
#               Prevents piling up stale utterances.
#
#  tts_queue   (Thread B → Thread C)
#    Payload : str (LLM reply text)
#    maxsize=1: Thread B blocks if Thread C is still speaking.
#               Natural turn-taking with no explicit lock.
#
#  Sentinel : None on a queue signals the consumer to exit cleanly.

audio_queue: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=1)
tts_queue:   "queue.Queue[Optional[str]]"        = queue.Queue(maxsize=1)

# Set by quit command (Thread B/D) or Ctrl+C (main) to stop all threads.
quit_event = threading.Event()

# Set by Thread C while the bot is speaking. Thread A discards VAD triggers
# while this is active so TTS output never re-enters the pipeline.
playback_active = threading.Event()

# Set by Thread D (Space key) to cut off the current bot utterance mid-speech.
# Thread C checks this in its playback loop and clears it after stopping.
stop_speaking_event = threading.Event()


# ─────────────────────  Load Models ─────────────────────────────

def load_all_models():
    """Load all models sequentially in the main thread before spawning workers."""
    print("Loading silero-vad (CPU)...")
    vad_model = load_silero_vad()

    print("Loading Whisper STT...")
    stt_model = whisper.load_model(STT_MODEL, device=DEVICE)

    print("Loading Qwen LLM...")
    tokenizer = AutoTokenizer.from_pretrained(LM_MODEL, trust_remote_code=True)
    lm_model  = AutoModelForCausalLM.from_pretrained(
        LM_MODEL,
        device_map="auto",
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        trust_remote_code=True,
    )

    print("Loading Coqui TTS...")
    tts_engine = TTS(TTS_MODEL).to(DEVICE)
    print("TTS speakers:", tts_engine.speakers)

    return vad_model, stt_model, lm_model, tokenizer, tts_engine


# ─────────────────────  Thread A — Capture ──────────────────────

def capture_thread(vad_model) -> None:
    """Stream mic input via VAD and push speech segments into audio_queue."""
    print("[Thread A] Started — waiting for speech.")

    while not quit_event.is_set():
        recorded_chunks: List[np.ndarray] = []
        silent_count       = 0
        speech_detected    = False
        speech_chunk_count = 0
        max_chunks         = int(MAX_RECORD_SECS * SAMPLE_RATE / CHUNK_SIZE)
        t_capture_start    = time.perf_counter()

        print("[Thread A] Listening...")

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=CHUNK_SIZE,
        ) as stream:
            for _ in range(max_chunks):

                if quit_event.is_set():
                    break

                chunk, _ = stream.read(CHUNK_SIZE)
                chunk_1d = chunk.squeeze()   # shape (CHUNK_SIZE,) float32

                # silero-vad expects a 1-D float32 tensor
                chunk_tensor = torch.from_numpy(chunk_1d)
                with torch.no_grad():
                    speech_prob = vad_model(chunk_tensor, SAMPLE_RATE).item()

                is_speech = speech_prob >= VAD_THRESHOLD

                if is_speech:
                    # Mute during bot playback — without AEC we cannot
                    # separate TTS output from a real user utterance.
                    if playback_active.is_set():
                        recorded_chunks.clear()
                        speech_detected    = False
                        speech_chunk_count = 0
                        silent_count       = 0
                        continue
                    speech_detected    = True
                    speech_chunk_count += 1
                    silent_count       = 0
                    recorded_chunks.append(chunk_1d)
                elif speech_detected:
                    # Only count silence after we've heard at least one speech chunk
                    if playback_active.is_set():
                        recorded_chunks.clear()
                        speech_detected    = False
                        speech_chunk_count = 0
                        silent_count       = 0
                        continue
                    silent_count += 1
                    recorded_chunks.append(chunk_1d)   # keep trailing silence
                    if silent_count >= SILENCE_CHUNKS:
                        break
                # Before any speech starts, keep listening (don't accumulate silence)

        if quit_event.is_set():
            break

        if not recorded_chunks:
            print("[Thread A] No speech in window — resuming.")
            continue

        # Noise gate — discard very short captures (fan noise, HVAC blips)
        if speech_chunk_count < MIN_SPEECH_CHUNKS:
            print(
                f"[Thread A] Too short ({speech_chunk_count} speech chunks) — "
                "discarding (likely noise)."
            )
            continue

        audio      = np.concatenate(recorded_chunks, axis=0)
        t_capture  = time.perf_counter() - t_capture_start
        audio_secs = len(audio) / SAMPLE_RATE
        print(
            f"[Timing - Thread A] Capture: {t_capture:.2f}s "
            f"({audio_secs:.2f}s of audio captured)"
        )

        audio_queue.put(audio)   # blocks if Thread B is still busy

    # Send poison pill to unblock Thread B
    try:
        audio_queue.put_nowait(None)
    except queue.Full:
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            pass
        audio_queue.put_nowait(None)

    print("[Thread A] Exiting.")


# ─────────────────────  Thread B — Inference ────────────────────

def inference_thread(stt_model, lm_model, tokenizer) -> None:
    """Transcribe audio with Whisper, generate a reply with Qwen, push to tts_queue."""
    chat_history: List[str] = []
    print("[Thread B] Started — waiting for audio.")

    while True:
        audio_np = audio_queue.get()

        if audio_np is None:
            print("[Thread B] Sentinel received — cascading to Thread C.")
            tts_queue.put(None)
            print("[Thread B] Exiting.")
            return

        # ──────────────────────── STT ───────────────────────────
        print("[Thread B] Transcribing...")
        t_stt_start = time.perf_counter()
        result = stt_model.transcribe(
            audio_np,
            language="en",
            fp16=(DEVICE == "cuda"),
        )
        text  = result["text"].strip()
        t_stt = time.perf_counter() - t_stt_start

        del audio_np   # free CPU RAM before LLM allocation

        print(f"[Thread B] You said: {text}")

        if not text:
            print("[Thread B] Empty transcription — resuming.")
            continue

        # Whisper sometimes returns internal special tokens (e.g. <|af|>)
        # on noisy clips. These are never valid user input; discard them.
        if "<|" in text:
            print("[Thread B] Hallucination filtered (special tokens) — resuming.")
            continue

        # Quit commands
        if text.lower().rstrip(".,!?") in ("quit", "exit", "stop"):
            print("[Thread B] Quit command detected — shutting down.")
            quit_event.set()
            stop_speaking_event.set()
            tts_queue.put("Goodbye!")
            tts_queue.put(None)
            print("[Thread B] Exiting.")
            return

        # ──────────────────────── LM ───────────────────────────
        print("[Thread B] Generating reply...")
        t_lm_start = time.perf_counter()
        reply = _generate_reply(text, chat_history, lm_model, tokenizer)
        t_lm  = time.perf_counter() - t_lm_start

        print(f"[Timing - Thread B] STT: {t_stt:.2f}s | LLM: {t_lm:.2f}s")

        tts_queue.put(reply)   # blocks if Thread C is still speaking


# ─────────────────────  Thread C — Output ───────────────────────

def output_thread(tts_engine) -> None:
    """Synthesise speech with Coqui TTS and play it back."""
    print("[Thread C] Started — waiting for text.")

    while True:
        text = tts_queue.get()

        if text is None:
            print("[Thread C] Sentinel received — exiting.")
            return

        # ──────────────────────── TTS ───────────────────────────
        print(f"Bot: {text}")
        tts_text    = _sanitize_for_tts(text)   # expand digits → words
        t_tts_start = time.perf_counter()
        wav         = tts_engine.tts(tts_text, language="en", speaker="female-en-5")
        t_tts       = time.perf_counter() - t_tts_start

        print(f"[Timing - Thread C] TTS: {t_tts:.2f}s")

        wav_array   = np.array(wav)
        sample_rate = tts_engine.synthesizer.output_sample_rate
        play_secs   = len(wav_array) / sample_rate

        stop_speaking_event.clear()   # reset before each new utterance
        playback_active.set()         # tell Thread A to mute mic capture
        try:
            sd.play(wav_array, samplerate=sample_rate)
            # Poll every 50 ms instead of blocking sd.wait() so that both
            # Ctrl+C (quit_event) and Space (stop_speaking_event) respond fast.
            t_play_start = time.perf_counter()
            interrupted  = False
            while time.perf_counter() - t_play_start < play_secs:
                if quit_event.is_set() or stop_speaking_event.is_set():
                    sd.stop()
                    interrupted = True
                    break
                time.sleep(0.05)
            if not interrupted:
                time.sleep(0.3)   # brief settling — absorbs mic tail ringing
        finally:
            playback_active.clear()

        del wav, wav_array   # free ~1–4 MB CPU RAM before next synthesis


# ─────────────────────  Thread D — Keyboard ─────────────────────

def keyboard_thread() -> None:
    """
    Poll for keypresses using Windows msvcrt (no pip install required).
    Space  — stop current bot speech and resume listening.
    q / Q  — graceful full shutdown (same as saying 'quit').
    """
    print("[Thread D] Keyboard ready   Space: stop speech | q: quit")

    while not quit_event.is_set():
        if msvcrt.kbhit():
            key = msvcrt.getwch()
            if key == " ":
                print("\n[Thread D] Space pressed — stopping speech.")
                stop_speaking_event.set()
            elif key in ("q", "Q"):
                print("\n[Thread D] Q pressed — quitting.")
                quit_event.set()
                stop_speaking_event.set()
                # Push sentinels so blocked workers unblock immediately
                for q in (audio_queue, tts_queue):
                    try:
                        q.put_nowait(None)
                    except queue.Full:
                        pass
                break
        time.sleep(0.05)

    print("[Thread D] Exiting.")


# ─────────────────────  TTS Pre-processing ──────────────────────

def _sanitize_for_tts(text: str) -> str:
    """
    Expand digit sequences to spoken words before passing text to Coqui TTS.
    The your_tts phoneme vocabulary has no digit characters; leaving them in
    causes silent drops and "Character 'X' not found" warnings.
    Handles integers 0 – 999 999 (years, ages, simple maths, dates).
    """
    def _int_to_words(n: int) -> str:
        if n < 0:
            return "negative " + _int_to_words(-n)
        if n == 0:
            return "zero"
        ones = ["", "one", "two", "three", "four", "five", "six",
                "seven", "eight", "nine", "ten", "eleven", "twelve",
                "thirteen", "fourteen", "fifteen", "sixteen",
                "seventeen", "eighteen", "nineteen"]
        tens = ["", "", "twenty", "thirty", "forty", "fifty",
                "sixty", "seventy", "eighty", "ninety"]
        if n < 20:
            return ones[n]
        if n < 100:
            return tens[n // 10] + ("-" + ones[n % 10] if n % 10 else "")
        if n < 1_000:
            tail = n % 100
            return ones[n // 100] + " hundred" + (
                " and " + _int_to_words(tail) if tail else "")
        if n < 1_000_000:
            high, low = divmod(n, 1_000)
            return _int_to_words(high) + " thousand" + (
                " " + _int_to_words(low) if low else "")
        return str(n)

    return re.sub(r"\b\d+\b", lambda m: _int_to_words(int(m.group())), text)


# ─────────────────────  Prompt Engineering ──────────────────────

def _build_chat_input(tokenizer, history: List[str], user_input: str):
    """
    Build tokenizer input from alternating [user, bot, user, bot, …] history.
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
    parts  = [f"SYSTEM: {system}", ""]
    for m in messages:
        parts.append(f"{m['role'].upper()}: {m['content'].strip()}")
    parts.append("ASSISTANT:")
    return "\n".join(parts)


# ─────────────────────  LM Response ─────────────────────────────

def _generate_reply(
    user_input: str,
    chat_history: List[str],
    lm,
    tokenizer,
) -> str:
    """Run Qwen inference and append the (user, reply) pair to chat_history."""
    prompt_text = _build_chat_input(tokenizer, chat_history, user_input)

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
        response = tokenizer.decode(
            output[0][input_len:], skip_special_tokens=True
        ).strip()
    except Exception:
        full     = tokenizer.decode(output[0], skip_special_tokens=True)
        response = full
        if isinstance(full, str) and full.startswith(prompt_text.strip()):
            response = full[len(prompt_text.strip()):].strip()

    chat_history.append(user_input)
    chat_history.append(response)
    if len(chat_history) > MAX_HISTORY:
        chat_history[:] = chat_history[-MAX_HISTORY:]

    return response


# ─────────────────────  Main ─────────────────────────────────────

def main() -> None:
    vad_model, stt_model, lm_model, tokenizer, tts_engine = load_all_models()

    print("\nVoice chatbot ready (Phase 3 — Multithreaded)")
    print("Say 'quit', 'exit', or 'stop' to end  |  Space: stop speech  |  q: quit")
    print()

    threads = [
        threading.Thread(
            target=capture_thread,
            args=(vad_model,),
            name="Thread-A-Capture",
            daemon=True,
        ),
        threading.Thread(
            target=inference_thread,
            args=(stt_model, lm_model, tokenizer),
            name="Thread-B-Inference",
            daemon=True,
        ),
        threading.Thread(
            target=output_thread,
            args=(tts_engine,),
            name="Thread-C-Output",
            daemon=True,
        ),
        threading.Thread(
            target=keyboard_thread,
            name="Thread-D-Keyboard",
            daemon=True,
        ),
    ]

    for t in threads:
        t.start()

    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\n[Main] Ctrl+C — signalling shutdown...")
        quit_event.set()
        stop_speaking_event.set()
        for q in (audio_queue, tts_queue):
            try:
                q.put_nowait(None)
            except queue.Full:
                pass

    print("[Main] All threads done. Exiting.")


if __name__ == "__main__":
    main()
