@echo off
cd /d C:\Users\zakar\homelab-gpu
set LOG=C:\Users\zakar\homelab-data\logs\reaction-train-accurate.log
set PYTHONUNBUFFERED=1
echo [%date% %time%] starting FER+ accurate train >> "%LOG%"
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run-reaction-train-accurate.ps1 >> "%LOG%" 2>&1
echo [%date% %time%] finished exit=%ERRORLEVEL% >> "%LOG%"
