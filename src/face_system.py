from __future__ import annotations

import csv
import pickle
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

try:
    from deepface import DeepFace
except ModuleNotFoundError:
    DeepFace = None


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / "dataset"
MODELS_DIR = ROOT_DIR / "models"
ATTENDANCE_DIR = ROOT_DIR / "attendance"
CLASSIFIER_PATH = MODELS_DIR / "face_classifier.pkl"
ATTENDANCE_PATH = ATTENDANCE_DIR / "attendance.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_DEEPFACE_MODEL = "Facenet"
DEEPFACE_INSTALL_HELP = (
    "The 'deepface' package is not installed in this Python environment. "
    "Install it with 'pip install deepface tf-keras'."
)


@dataclass
class RecognitionResult:
    name: str
    confidence: float
    location: Tuple[int, int, int, int]


def ensure_project_dirs() -> None:
    """Create runtime folders if they do not exist."""
    DATASET_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    ATTENDANCE_DIR.mkdir(exist_ok=True)


def require_deepface() -> None:
    if DeepFace is None:
        raise ModuleNotFoundError(DEEPFACE_INSTALL_HELP)


def clean_person_name(name: str) -> str:
    safe = "".join(ch for ch in name.strip() if ch.isalnum() or ch in (" ", "_", "-"))
    return "_".join(safe.split())


def iter_dataset_images(dataset_dir: Path = DATASET_DIR) -> Iterable[Tuple[str, Path]]:
    for person_dir in sorted(dataset_dir.iterdir() if dataset_dir.exists() else []):
        if not person_dir.is_dir():
            continue
        for image_path in sorted(person_dir.rglob("*")):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                yield person_dir.name, image_path


def load_rgb_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def detect_faces(frame_bgr: np.ndarray, resize_scale: float = 1.0) -> List[Tuple[int, int, int, int]]:
    """Detect faces with OpenCV Haar cascade and return locations as top, right, bottom, left."""
    if frame_bgr is None or frame_bgr.size == 0:
        return []

    scale = min(max(resize_scale, 0.2), 1.0)
    small_frame = cv2.resize(frame_bgr, (0, 0), fx=scale, fy=scale) if scale != 1.0 else frame_bgr
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    locations: List[Tuple[int, int, int, int]] = []
    for x, y, width, height in faces:
        left = int(x / scale)
        top = int(y / scale)
        right = int((x + width) / scale)
        bottom = int((y + height) / scale)
        locations.append((top, right, bottom, left))
    return locations


def crop_face(frame_bgr: np.ndarray, location: Tuple[int, int, int, int], padding: float = 0.18) -> np.ndarray:
    top, right, bottom, left = location
    height, width = frame_bgr.shape[:2]
    box_width = right - left
    box_height = bottom - top
    pad_x = int(box_width * padding)
    pad_y = int(box_height * padding)
    left = max(0, left - pad_x)
    right = min(width, right + pad_x)
    top = max(0, top - pad_y)
    bottom = min(height, bottom + pad_y)
    return frame_bgr[top:bottom, left:right]


def deepface_embedding(image_bgr: np.ndarray, model_name: str = DEFAULT_DEEPFACE_MODEL) -> np.ndarray:
    require_deepface()
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty face image.")

    # DeepFace accepts numpy images. enforce_detection=False is intentional because
    # this project already detects/crops faces with OpenCV for speed and stability.
    representation = DeepFace.represent(
        img_path=image_bgr,
        model_name=model_name,
        detector_backend="skip",
        enforce_detection=False,
    )
    embedding = representation[0]["embedding"] if isinstance(representation, list) else representation["embedding"]
    return np.asarray(embedding, dtype=np.float32)


def extract_embeddings(
    dataset_dir: Path = DATASET_DIR,
    model_name: str = DEFAULT_DEEPFACE_MODEL,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract DeepFace embeddings for every image in dataset/person folders."""
    require_deepface()
    embeddings: List[np.ndarray] = []
    labels: List[str] = []
    skipped: List[str] = []

    for person_name, image_path in iter_dataset_images(dataset_dir):
        try:
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                skipped.append(f"{image_path} - could not read image")
                continue

            locations = detect_faces(image_bgr)
            if not locations:
                # If the file is already a cropped face, still try to embed it.
                face_image = image_bgr
            else:
                face_image = crop_face(image_bgr, locations[0])

            # Use the first face for training images. Keep training images one-person-per-frame.
            embeddings.append(deepface_embedding(face_image, model_name=model_name))
            labels.append(person_name)
        except Exception as exc:
            skipped.append(f"{image_path} - {exc}")

    return np.asarray(embeddings), np.asarray(labels), skipped


def train_classifier(
    dataset_dir: Path = DATASET_DIR,
    model_path: Path = CLASSIFIER_PATH,
    classifier_type: str = "svm",
    model_name: str = DEFAULT_DEEPFACE_MODEL,
) -> dict:
    ensure_project_dirs()
    embeddings, labels, skipped = extract_embeddings(dataset_dir, model_name=model_name)

    if len(embeddings) < 2:
        raise ValueError("Add at least two usable face images before training.")
    if len(set(labels)) < 2:
        raise ValueError("Add images for at least two different people before training.")

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    if classifier_type.lower() == "knn":
        classifier = KNeighborsClassifier(n_neighbors=min(3, len(embeddings)))
    else:
        classifier = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("svm", SVC(kernel="linear", probability=True, class_weight="balanced")),
            ]
        )

    stratify = encoded_labels if min(np.bincount(encoded_labels)) > 1 else None
    accuracy: Optional[float] = None

    if len(embeddings) >= 6 and stratify is not None:
        x_train, x_test, y_train, y_test = train_test_split(
            embeddings,
            encoded_labels,
            test_size=0.25,
            random_state=42,
            stratify=stratify,
        )
        classifier.fit(x_train, y_train)
        accuracy = float(accuracy_score(y_test, classifier.predict(x_test)))
        classifier.fit(embeddings, encoded_labels)
    else:
        classifier.fit(embeddings, encoded_labels)

    payload = {
        "classifier": classifier,
        "label_encoder": label_encoder,
        "classes": list(label_encoder.classes_),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "embedding_count": int(len(embeddings)),
        "accuracy": accuracy,
        "classifier_type": classifier_type,
        "deepface_model": model_name,
    }

    with model_path.open("wb") as file:
        pickle.dump(payload, file)

    return {**payload, "skipped": skipped, "model_path": str(model_path)}


def load_model(model_path: Path = CLASSIFIER_PATH) -> dict:
    if not model_path.exists():
        raise FileNotFoundError("No trained model found. Train the model first.")
    with model_path.open("rb") as file:
        return pickle.load(file)


def recognize_faces(
    frame_bgr: np.ndarray,
    model_payload: dict,
    tolerance: float = 0.55,
    resize_scale: float = 0.5,
) -> List[RecognitionResult]:
    """Recognize all faces in a BGR OpenCV frame."""
    require_deepface()
    if frame_bgr is None or frame_bgr.size == 0:
        return []

    locations = detect_faces(frame_bgr, resize_scale=resize_scale)
    classifier = model_payload["classifier"]
    label_encoder: LabelEncoder = model_payload["label_encoder"]
    model_name = model_payload.get("deepface_model", DEFAULT_DEEPFACE_MODEL)
    results: List[RecognitionResult] = []

    for location in locations:
        try:
            face_image = crop_face(frame_bgr, location)
            embedding = deepface_embedding(face_image, model_name=model_name)
            probabilities = classifier.predict_proba([embedding])[0]
            best_index = int(np.argmax(probabilities))
            confidence = float(probabilities[best_index])
            name = str(label_encoder.inverse_transform([best_index])[0])
            if confidence < tolerance:
                name = "Unknown"
            results.append(RecognitionResult(name=name, confidence=confidence, location=location))
        except Exception:
            results.append(RecognitionResult(name="Unknown", confidence=0.0, location=location))

    return results


def draw_results(frame_bgr: np.ndarray, results: Sequence[RecognitionResult], fps: Optional[float] = None) -> np.ndarray:
    output = frame_bgr.copy()

    for result in results:
        top, right, bottom, left = result.location
        known = result.name != "Unknown"
        color = (64, 220, 124) if known else (80, 120, 255)
        label = f"{result.name} ({result.confidence * 100:.1f}%)"

        cv2.rectangle(output, (left, top), (right, bottom), color, 2)
        cv2.rectangle(output, (left, max(0, top - 32)), (right, top), color, cv2.FILLED)
        cv2.putText(output, label, (left + 6, max(20, top - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (15, 15, 15), 2)

    if fps is not None:
        cv2.putText(output, f"FPS: {fps:.1f}", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return output


def mark_attendance(name: str, attendance_path: Path = ATTENDANCE_PATH) -> bool:
    """Write one attendance row per person per day. Returns True when a new row is added."""
    if not name or name == "Unknown":
        return False

    ensure_project_dirs()
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    existing = set()

    if attendance_path.exists():
        with attendance_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                existing.add((row.get("Name"), row.get("Date")))

    if (name, today) in existing:
        return False

    file_exists = attendance_path.exists()
    with attendance_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["Name", "Date", "Time"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({"Name": name, "Date": today, "Time": now.strftime("%H:%M:%S")})

    return True


def capture_person_images(
    person_name: str,
    count: int = 30,
    camera_index: int = 0,
    delay_seconds: float = 0.15,
) -> Path:
    """Capture face images from webcam for a person. Press q in the camera window to stop early."""
    ensure_project_dirs()
    safe_name = clean_person_name(person_name)
    if not safe_name:
        raise ValueError("Person name cannot be empty.")

    person_dir = DATASET_DIR / safe_name
    person_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions or camera index.")

    saved = 0
    try:
        while saved < count:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Could not read frame from webcam.")

            locations = detect_faces(frame)

            for top, right, bottom, left in locations[:1]:
                face = crop_face(frame, (top, right, bottom, left), padding=0.2)
                if face.size == 0:
                    continue
                saved += 1
                cv2.imwrite(str(person_dir / f"{safe_name}_{saved:03d}.jpg"), face)
                cv2.rectangle(frame, (left, top), (right, bottom), (64, 220, 124), 2)
                break

            cv2.putText(frame, f"Saved: {saved}/{count}", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Dataset Collection - press q to stop", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            time.sleep(delay_seconds)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return person_dir
