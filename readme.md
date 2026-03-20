# Modular Edge Voice Chatbot

**Master's Thesis Project - University of Vaasa**

This repository contains the implementation of a fully local, edge-deployed conversational AI pipeline. The system integrates Speech-to-Text (STT), a Large Language Model (LLM), and Text-to-Speech (TTS) to create a low-latency voice assistant capable of running on consumer-grade hardware (tested on an NVIDIA RTX 3060 6GB).

## Repository Contents

| Script | Phase | Description |
|---|---|---|
| `chatbot-dialogpt.py` | Legacy | Lightweight baseline using DialoGPT |
| `chatbot-moderate-qwen.py` | **Phase 1** | Baseline sequential pipeline: hardcoded 5-second recording window. Whisper STT + Qwen 2.5 (1.5B) + Coqui TTS |
| `chatbot-vad-qwen.py` | **Phase 2** | Adaptive listening with `silero-vad`: recording stops dynamically after ~640 ms of silence. Adds per-turn timing logs for thesis runtime analysis |
| `requirements.txt` | — | Locked, cross-platform dependencies with GPU acceleration |

---

## Hardware & OS Requirements
* **OS:** Windows 10/11 or Linux (Ubuntu)
* **GPU:** NVIDIA GPU with at least 6GB VRAM (CUDA 12.1 supported)
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
* In either script, confirm: `LM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"`

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

*Note: The first run will download Whisper, Coqui, and Qwen models from Hugging Face (~several minutes).*

**To exit:** Say **"Quit"**, **"Exit"**, or **"Stop"** — or press **Ctrl + C**.

---

## Project Phases & Status

| Phase | Script | Description | Status |
|---|---|---|---|
| Phase 1 | `chatbot-moderate-qwen.py` | Baseline sequential pipeline | ✅ Complete |
| Phase 2 | `chatbot-vad-qwen.py` | Adaptive listening with `silero-vad` | ✅ Complete |
| Phase 3 | *(TBD)* | Multithreaded concurrent pipeline | 🔲 Planned |

### Phase 2 — Technical Notes

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
These logs are used directly for thesis runtime analysis and comparison across phases.