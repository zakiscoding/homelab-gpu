@echo off
cd /d C:\Users\zakar\homelab-gpu
set LOG=C:\Users\zakar\homelab-data\logs\reaction-train.log
set PYTHONUNBUFFERED=1
echo [%date% %time%] starting FER2013 train >> "%LOG%"
C:\Users\zakar\homelab-gpu\.venv\Scripts\python.exe -u scripts\train_reaction_gpu.py --dataset fer2013 --epochs 30 --batch-size 64 >> "%LOG%" 2>&1
echo [%date% %time%] finished exit=%ERRORLEVEL% >> "%LOG%"
