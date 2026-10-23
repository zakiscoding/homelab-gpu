"""Webcam inference for reaction ResNet checkpoint (best_model.pt)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    classes = ckpt.get("classes") or ckpt.get("meta", {}).get("classes", [])
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    return model, classes


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    p.add_argument("--threshold", type=float, default=0.45)
    p.add_argument("--camera", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = load_model(Path(args.checkpoint), device)
    tf = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cap = cv2.VideoCapture(args.camera)
    print("q to quit | classes:", classes)

    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(48, 48))
            label, conf, reason = "—", 0.0, "no_face"
            if len(faces):
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                roi = frame[y : y + h, x : x + w]
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                t = tf(roi_rgb).unsqueeze(0).to(device)
                probs = torch.softmax(model(t), dim=1)[0]
                conf = float(probs.max())
                idx = int(probs.argmax())
                if conf >= args.threshold:
                    label = classes[idx]
                    reason = "fer_classifier"
                else:
                    label = "ambiguous"
                    reason = "ambiguous_expression"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"{label} {conf:.2f} ({reason})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.imshow("reaction-infer", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
