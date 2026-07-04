> ## Project Status
>
> This repository contains the original implementation developed as part of my Master's thesis at the University of Vaasa.
>
> Active development now continues in the successor project:
>
> **➡️ Edge Voice Assistant**
>
> https://github.com/FahadiF/Edge-Voice-Assistant
>
> Edge Voice Assistant builds upon the ideas explored in this research while introducing a redesigned architecture aimed at production-quality offline voice interaction, with:
>
> - Streaming voice interaction
> - True voice interruption (barge-in)
> - Modular architecture
> - Modern model management
> - Improved developer experience
> - Desktop and web interfaces
> - Production-quality engineering
>
> This repository is preserved as the research reference implementation accompanying the Master's thesis. Keeping the repositories separate preserves the exact implementation evaluated in the thesis while allowing the successor project to evolve independently.

---

# Modular-Software-Implementation-Edge-Voice-Chatbot

**Master's Thesis Project - University of Vaasa**

This repository contains the implementation of a fully local, edge-deployed conversational AI pipeline. The system integrates Speech-to-Text (STT), a Large Language Model (LLM), and Text-to-Speech (TTS) to create a low-latency voice assistant capable of running on consumer-grade hardware (tested on an NVIDIA RTX 3060 6GB).

This research has been done in conjunction with the Human–AI Interaction and Innovation Nexus at Vaasa.

## Project Evolution

| Period | Milestone |
|---|---|
| 2025–2026 | 🎓 **Master's Thesis** — Modular Software Implementation – Edge Voice Chatbot |
| 2026– | 🚀 **Edge Voice Assistant** — Production-grade offline AI voice assistant |

## Repository Contents

| Script | Phase | Description |
|---|---|---|
| `chatbot-dialogpt.py` | Legacy | Lightweight baseline using DialoGPT |
| `chatbot-moderate-qwen.py` | **Phase 1** | Baseline sequential pipeline: hardcoded 5-second recording window. Whisper STT + Qwen 2.5 (1.5B) + Coqui TTS |
| `chatbot-vad-qwen.py` | **Phase 2** | Adaptive listening with `silero-vad`: recording stops dynamically after ~640 ms of silence. Adds per-turn timing logs for thesis runtime analysis |
| `chatbot-threaded-qwen.py` | **Phase 3** | Concurrent multithreaded pipeline: VAD capture, Whisper + Qwen inference, and Coqui TTS run in parallel threads. Keyboard interrupt, echo suppression, and optional headset voice barge-in |
| `requirements.txt` | — | Locked, cross-platform dependencies with GPU acceleration |

---

## Hardware & OS Requirements
* **OS:** Windows 10/11 or Linux (Ubuntu)
* **GPU:** NVIDIA GPU with at least 6 GB VRAM (CUDA 12.1 supported)
* **Audio:** Functioning default microphone and speakers

---

## Installation & Setup

**1. Create and Activate a Virtual Environment**
```bash
python -m venv edge_env

# On Windows:
edge_env\Scripts\activate
# On Linux/macOS:
source edge_env/bin/activate
```

**2. Install Dependencies**
```bash
pip install -r requirements.txt
```

**3. Windows Note (AWQ Bypass)**
The `autoawq` library requires `triton`, which is Linux-only. On Windows, use the standard 1.5B model:
* In any script, confirm: `LM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"`

---

## Running the Chatbot

Ensure your virtual environment is active and microphone is unmuted.

**Phase 1 — Baseline (fixed 5-second window):**
```bash
python chatbot-moderate-qwen.py
```

**Phase 2 — Adaptive Listening (VAD):**
```bash
python chatbot-vad-qwen.py
```

**Phase 3 — Multithreaded Pipeline:**
```bash
python chatbot-threaded-qwen.py
```

*Note: The first run will download Whisper, Coqui, and Qwen models from Hugging Face (~several minutes).*

**Controls (Phase 3):**

| Action | How |
|---|---|
| End session | Say **"Quit"**, **"Exit"**, or **"Stop"** |
| Stop current speech | Press **Space** |
| Quit immediately | Press **q** |
| Emergency exit | **Ctrl + C** |

*Keyboard controls are cross-platform: `msvcrt` on Windows; `termios` + `select` on Linux/macOS. No additional pip package is needed on either OS.*

---

## Project Phases & Status

| Phase | Script | Description | Status |
|---|---|---|---|
| Phase 1 | `chatbot-moderate-qwen.py` | Baseline sequential pipeline | ✅ Complete |
| Phase 2 | `chatbot-vad-qwen.py` | Adaptive listening with `silero-vad` | ✅ Complete |
| Phase 3 | `chatbot-threaded-qwen.py` | Multithreaded concurrent pipeline | ✅ Complete |

---

## Technical Notes

### Phase 2 — VAD Adaptive Listening

**VAD Library:** `silero-vad` (neural, ~1.5 MB model, runs on CPU — preserves VRAM for Whisper + LLM).

**Key Parameters** (tunable at the top of `chatbot-vad-qwen.py`):

| Parameter | Default | Effect |
|---|---|---|
| `VAD_THRESHOLD` | `0.5` | Speech probability cutoff (0–1) |
| `SILENCE_CHUNKS` | `20` | ~640 ms of silence triggers stop |
| `MAX_RECORD_SECS` | `15` | Hard recording timeout (safety net) |
| `CHUNK_SIZE` | `512` | Samples per VAD inference call (~32 ms) |

**STT Improvement:** Phase 2 uses `whisper.transcribe()` instead of the low-level `decode()` call from Phase 1. This correctly handles variable-length audio and eliminates hallucinations caused by padding silence.

**Timing Logs:** Each turn prints:
```
[Timing] Record: Xs | STT: Xs | LM: Xs | TTS: Xs | Total: Xs
```

---

### Phase 3 — Multithreaded Pipeline

**Architecture:** Three worker threads communicate via `queue.Queue(maxsize=1)`:

```
Thread A (Capture)  →[audio_queue]→  Thread B (Inference)  →[tts_queue]→  Thread C (Output)
   VAD + mic                           Whisper + Qwen                        Coqui + playback
Thread D (Keyboard)  — Space: stop speech | q: quit
```

**Key Parameters** (tunable at the top of `chatbot-threaded-qwen.py`):

| Parameter | Default | Effect |
|---|---|---|
| `STT_MODEL` | `"base"` | Whisper model size: `tiny` / `base` / `small` / `medium` / `large` |
| `VAD_THRESHOLD` | `0.5` | Speech probability cutoff (0–1) |
| `SILENCE_CHUNKS` | `30` | ~960 ms of silence triggers stop |
| `MIN_SPEECH_CHUNKS` | `12` | ~384 ms minimum to pass noise gate |
| `MAX_RECORD_SECS` | `15` | Hard recording timeout (safety net) |
| `HEADSET_MODE` | `False` | Enable voice barge-in when using a headset (see below) |
| `BARGE_IN_CHUNKS` | `20` | ~640 ms of sustained speech required to trigger barge-in |

**Echo Suppression & HEADSET_MODE:**

Without hardware Acoustic Echo Cancellation (AEC), the microphone on a laptop picks up the bot's own TTS output through the speakers, causing the VAD to fire on the bot's voice and interrupt it mid-sentence.

The `HEADSET_MODE` constant controls how this is handled:

| `HEADSET_MODE` | Behaviour |
|---|---|
| `False` *(default)* | Mic capture is muted during playback. The bot never interrupts itself. Use Space to stop speech manually. |
| `True` | Mic stays open during playback. After `BARGE_IN_CHUNKS` (~640 ms) of sustained user speech, the bot's current utterance is cut off and Thread A immediately starts capturing the user's new sentence. Use this only with headphones or an external mic placed away from the speakers. |

**STT Accuracy:** The default `STT_MODEL = "base"` is optimised for speed within the 6 GB VRAM budget. Switching to `"small"` (~+320 MB VRAM) or `"medium"` (~+1.5 GB VRAM) noticeably improves transcription accuracy at the cost of slightly higher latency.

**LLM Output Length:** The system prompt instructs the model to answer in one to two short sentences, and `max_new_tokens` is capped at 150. This combination ensures replies finish naturally without mid-sentence clipping while keeping spoken output concise.

**Keyboard Interrupt (Thread D):** Cross-platform, no pip install required. Polls every 50 ms.

| OS | Library used |
|---|---|
| Windows | `msvcrt` (built-in) |
| Linux / macOS | `termios` + `select` (built-in) |

**Timing Logs:** Each thread prints independently:
```
[Timing - Thread A] Capture: Xs (Xs of audio captured)
[Timing - Thread B] STT: Xs | LLM: Xs
[Timing - Thread C] TTS: Xs
```

**Steady-state VRAM usage:**

| Model | VRAM |
|---|---|
| Whisper base | ~145 MB |
| Qwen 2.5-1.5B (fp16) | ~3.0 GB |
| Coqui TTS | ~150 MB |
| **Total** | **~3.3 GB** |
