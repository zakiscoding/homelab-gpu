# Capture webcam frames into class folders for personal reaction training.
param(
    [string]$OutRoot = "C:\Users\zakar\homelab-data\datasets\me-reactions\train",
    [string[]]$Classes = @("neutral", "happy", "surprise", "sad", "angry")
)

$ErrorActionPreference = "Stop"
$Venv = Join-Path (Split-Path (Split-Path $MyInvocation.MyCommand.Path -Parent) -Parent) ".venv\Scripts\python.exe"
if (-not (Test-Path $Venv)) { $Venv = "python" }

$py = @'
import argparse, os, time
from datetime import datetime
from pathlib import Path
import cv2

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--classes", nargs="+", required=True)
    args = p.parse_args()
    root = Path(args.out)
    for c in args.classes:
        (root / c).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Webcam not available")
    idx = 0
    current = args.classes[0]
    print("Keys: 0-9 select class index | SPACE save | q quit")
    print("Classes:", list(enumerate(args.classes)))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.putText(frame, f"class={current} saved={idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("capture-reactions", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if ord("0") <= k <= ord("9"):
            i = k - ord("0")
            if i < len(args.classes):
                current = args.classes[i]
        if k == ord(" "):
            path = root / current / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
            cv2.imwrite(str(path), frame)
            idx += 1
            print("saved", path)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
'@

$tmp = Join-Path $env:TEMP "capture_reactions.py"
Set-Content -Path $tmp -Value $py -Encoding UTF8
& $Venv $tmp --out $OutRoot --classes @($Classes)
