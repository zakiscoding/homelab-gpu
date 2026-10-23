@echo off
set REACTION_CHECKPOINT=C:\Users\zakar\homelab-data\checkpoints\8117bfe1-85ea-4074-a967-5ebeedd40698\best_model.pt
cd /d C:\Users\zakar\homelab-gpu
C:\Users\zakar\homelab-gpu\.venv\Scripts\python.exe -m uvicorn reaction.serve:app --host 0.0.0.0 --port 8100 >> C:\Users\zakar\homelab-data\logs\reaction-serve.log 2>&1
