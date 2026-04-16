"""Run mask detection on an image, a folder, or a webcam stream."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch
from PIL import Image

from src.mask_detection.constants import CLASS_COLORS
from src.mask_detection.runtime import (
    build_model_from_checkpoint,
    load_checkpoint,
    preprocess_pil_image,
    select_device,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect face-mask status with OpenCV.")
    parser.add_argument(
        "--checkpoint",
        default="artifacts/best_model.pt",
        help="Path to the trained PyTorch checkpoint.",
    )
    parser.add_argument("--image", help="Path to a single image file.", default=None)
    parser.add_argument("--input-dir", help="Directory of images to annotate.", default=None)
    parser.add_argument("--webcam", action="store_true", help="Use the default webcam.")
    parser.add_argument(
        "--output-dir",
        default="predictions",
        help="Directory where annotated outputs will be written.",
    )
    return parser


def load_face_detector() -> cv2.CascadeClassifier:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(f"Unable to load Haar cascade from {cascade_path}")
    return detector


@torch.no_grad()
def classify_face(face_bgr, model, class_names, image_size, device):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    tensor = preprocess_pil_image(Image.fromarray(face_rgb), image_size=image_size).to(device)
    logits = model(tensor)
    probabilities = torch.softmax(logits, dim=1)[0]
    class_index = int(torch.argmax(probabilities).item())
    return class_names[class_index], float(probabilities[class_index].item())


def annotate_frame(frame, detector, model, class_names, image_size, device):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    for (x, y, w, h) in faces:
        face = frame[y : y + h, x : x + w]
        if face.size == 0:
            continue
        label, confidence = classify_face(face, model, class_names, image_size, device)
        color = CLASS_COLORS.get(label, (255, 255, 255))
        text = f"{label} {confidence:.2f}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            frame,
            text,
            (x, max(25, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return frame


def process_image_file(image_path: Path, output_dir: Path, detector, model, class_names, image_size, device):
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"Skipping unreadable image: {image_path}")
        return
    annotated = annotate_frame(frame, detector, model, class_names, image_size, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / image_path.name
    cv2.imwrite(str(output_path), annotated)
    print(f"Saved: {output_path}")


def run_webcam(detector, model, class_names, image_size, device):
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise RuntimeError("Unable to open the default webcam.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            annotated = annotate_frame(frame, detector, model, class_names, image_size, device)
            cv2.imshow("Mask Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main() -> None:
    args = build_parser().parse_args()
    if not any([args.image, args.input_dir, args.webcam]):
        raise SystemExit("Provide --image, --input-dir, or --webcam.")

    device = select_device()
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    model = build_model_from_checkpoint(checkpoint).to(device)
    model.eval()
    class_names = checkpoint["class_names"]
    image_size = int(checkpoint["image_size"])
    detector = load_face_detector()
    output_dir = Path(args.output_dir)

    if args.image:
        process_image_file(Path(args.image), output_dir, detector, model, class_names, image_size, device)

    if args.input_dir:
        for image_path in sorted(Path(args.input_dir).glob("*")):
            if image_path.is_file():
                process_image_file(image_path, output_dir, detector, model, class_names, image_size, device)

    if args.webcam:
        run_webcam(detector, model, class_names, image_size, device)


if __name__ == "__main__":
    main()
