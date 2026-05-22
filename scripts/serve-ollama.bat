@echo off
set OLLAMA_HOST=0.0.0.0:11434
"%LOCALAPPDATA%\Programs\Ollama\ollama.exe" serve
