# Modular Edge Voice Chatbot

**Master's Thesis Project - University of Vaasa**

Currently contains the baseline implementation files provided by Prof. Boutellier.
This repository contains the implementation of a fully local, edge-deployed conversational AI pipeline. The system integrates Speech-to-Text (STT), a Large Language Model (LLM), and Text-to-Speech (TTS) to create a low-latency voice assistant capable of running on consumer-grade hardware (tested on an NVIDIA RTX 3060 6GB).

## Repository Contents
* `chatbot-dialogpt.py` - Lightweight baseline implementation using DialoGPT.
* `chatbot-moderate-qwen.py` - Advanced implementation utilizing Qwen 2.5 (1.5B/3B), Whisper STT, and Coqui TTS.
* `requirements.txt` - Locked, cross-platform dependencies ensuring GPU acceleration.

---

## Hardware & OS Requirements
* **OS:** Windows 10/11 or Linux (Ubuntu)
* **GPU:** NVIDIA GPU with at least 6GB VRAM (CUDA 12.1 supported)
* **Audio:** Functioning default microphone and speakers

---

## Installation & Setup

**1. Create and Activate a Virtual Environment**
To prevent dependency conflicts, it is highly recommended to use an isolated Python virtual environment.
```bash
python -m venv edge_env

# On Windows:
edge_env\Scripts\activate
# On Linux/macOS:
source edge_env/bin/activate
```

**2. Install Dependencies**
Install the exact required packages. The `requirements.txt` is pre-configured to fetch the GPU-accelerated version of PyTorch automatically.
```bash
pip install -r requirements.txt
```

**3. The Windows AWQ Bypass (Windows Users Only)**
The baseline script relies on `autoawq` to run a quantized 3B Qwen model. However, `autoawq` requires `triton`, which is strictly Linux-only. If you are running this on Windows, you must use the standard PyTorch 1.5B model instead:
* Open `chatbot-moderate-qwen.py`.
* Change Line 46 to: `LM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"`

---

## Running the Chatbot
Ensure your virtual environment is active and your microphone is unmuted, then run:

```bash
python chatbot-moderate-qwen.py
```
*Note: The first run will take several minutes to download the Whisper, Coqui, and Qwen models from Hugging Face to your local cache.*

**To exit the application:** Wait for the terminal to print `Listening...` and simply say **"Quit"** or press **"Ctrl + C"** on keyboard to force exit.

---

## Current Status & Next Steps

**Current Status (Phase 1 Complete):** The baseline sequential pipeline (Whisper STT -> Qwen 1.5B LLM -> Coqui TTS) is successfully running locally on Windows 11 with full RTX 3060 hardware acceleration.

**Identified Issue:** The current script utilizes a hardcoded 5-second recording loop (`sd.rec(duration=5)`). This forces the Whisper model to transcribe ambient room silence when the user is not actively speaking, resulting in severe AI hallucinations (e.g., transcribing repeating numbers or gibberish). This pollutes the LLM's context window and eventually causes Out-of-Memory (OOM) crashes.

**Phase 2 Objective (Adaptive Listening):** Implement a lightweight Voice Activity Detection (VAD) module (e.g., WebRTC VAD or Silero) to replace the hardcoded recording window. The system will be refactored to only capture and pass audio to Whisper when human speech is actively detected, terminating the recording dynamically after a brief period of silence.