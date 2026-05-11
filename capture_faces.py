from __future__ import annotations

import argparse

from src.face_system import capture_person_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture face images for a new person.")
    parser.add_argument("--name", required=True, help="Person name. A folder will be created under dataset/.")
    parser.add_argument("--count", type=int, default=30, help="Number of face images to save.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        person_dir = capture_person_images(args.name, count=args.count, camera_index=args.camera)
        print(f"Images saved in: {person_dir}")
    except Exception as exc:
        raise SystemExit(f"Capture failed: {exc}") from exc


if __name__ == "__main__":
    main()
