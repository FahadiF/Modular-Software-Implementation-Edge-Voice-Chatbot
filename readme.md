# Modular Edge Voice Chatbot

**Master's Thesis Project — University of Vaasa**

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