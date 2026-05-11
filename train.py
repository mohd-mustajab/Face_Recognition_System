from __future__ import annotations

import argparse

from src.face_system import train_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the face recognition classifier.")
    parser.add_argument("--classifier", choices=["svm", "knn"], default="svm", help="Classifier to train on face embeddings.")
    parser.add_argument(
        "--model",
        choices=["Facenet", "VGG-Face", "ArcFace", "OpenFace", "DeepFace"],
        default="Facenet",
        help="DeepFace embedding model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = train_classifier(classifier_type=args.classifier, model_name=args.model)
        print(f"Model saved: {result['model_path']}")
        print(f"People: {', '.join(result['classes'])}")
        print(f"Embeddings: {result['embedding_count']}")
        print(f"DeepFace model: {result['deepface_model']}")
        if result["accuracy"] is not None:
            print(f"Validation accuracy: {result['accuracy'] * 100:.2f}%")
        if result["skipped"]:
            print("\nSkipped images:")
            for item in result["skipped"]:
                print(f"- {item}")
    except Exception as exc:
        raise SystemExit(f"Training failed: {exc}") from exc


if __name__ == "__main__":
    main()
