@echo off
cd /d C:\Users\zakar\homelab-gpu
set LOG=C:\Users\zakar\homelab-data\logs\reaction-accurate.log
set PYTHONUNBUFFERED=1
set CNN_NUM_WORKERS=10
set CNN_PREFETCH=4
set TRAIN_MAIN_THREADS=4
set OMP_NUM_THREADS=1
echo [%date% %time%] starting FER ImageFolder ResNet34 train >> "%LOG%"
C:\Users\zakar\homelab-gpu\.venv\Scripts\python.exe -u scripts\train_reaction_gpu.py --dataset folder --dataset-version fer2013 --arch resnet34 --epochs 60 --batch-size 48 --grad-accum 4 --lr 3e-4 --label-smoothing 0.1 --class-weights --accurate-aug >> "%LOG%" 2>&1
echo [%date% %time%] finished exit=%ERRORLEVEL% >> "%LOG%"
