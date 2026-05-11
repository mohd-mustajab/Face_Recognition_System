from __future__ import annotations

import argparse
import time

import cv2

from src.face_system import draw_results, load_model, mark_attendance, recognize_faces


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time face recognition from webcam.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--threshold", type=float, default=0.55, help="Minimum prediction confidence for known names.")
    parser.add_argument("--scale", type=float, default=0.5, help="Frame resize scale for faster processing.")
    parser.add_argument("--no-attendance", action="store_true", help="Disable automatic attendance marking.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_payload = load_model()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Check camera permissions or camera index.")

    previous_time = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise SystemExit("Could not read from webcam.")

            results = recognize_faces(frame, model_payload, tolerance=args.threshold, resize_scale=args.scale)
            if not args.no_attendance:
                for result in results:
                    mark_attendance(result.name)

            now = time.time()
            fps = 1.0 / max(now - previous_time, 1e-6)
            previous_time = now

            cv2.imshow("Face Recognition - press q to quit", draw_results(frame, results, fps=fps))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
