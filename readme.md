# Edge Voice Chatbot (Master's Thesis)

Initial repository for my Master's Thesis project at the University of Vaasa.

Currently contains the baseline implementation files provided by Prof. Boutellier:

* `chatbot-dialogpt.py` (DialoGPT implementation)
* `chatbot-moderate-qwen.py` (Qwen implementation)

**Setup notes:**
Basic requirements needed to run the baseline models:

* torch, torchaudio, torchvision
* transformers
* openai-whisper
* TTS
* autoawq
* accelerate

Next steps: Test baseline scripts on local RTX 3060 and begin planning the modular architecture.
