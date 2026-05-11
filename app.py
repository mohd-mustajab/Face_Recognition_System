from __future__ import annotations

import tempfile
import time
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from src.face_system import (
    ATTENDANCE_PATH,
    CLASSIFIER_PATH,
    DATASET_DIR,
    DEEPFACE_INSTALL_HELP,
    DEFAULT_DEEPFACE_MODEL,
    DeepFace,
    capture_person_images,
    draw_results,
    ensure_project_dirs,
    load_model,
    mark_attendance,
    recognize_faces,
    train_classifier,
)


st.set_page_config(
    page_title="Face Recognition Attendance System",
    page_icon=":camera:",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    .stApp {
        background: radial-gradient(circle at top left, #18243a 0, #0d1117 34%, #080b10 100%);
        color: #f8fafc;
    }
    [data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid rgba(148, 163, 184, 0.18);
    }
    .hero {
        padding: 1.25rem 0 0.75rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.2);
        margin-bottom: 1rem;
    }
    .metric-box {
        background: rgba(15, 23, 42, 0.76);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 8px;
        padding: 1rem;
    }
    .small-muted {
        color: #94a3b8;
        font-size: 0.92rem;
    }
    .success-text {
        color: #86efac;
        font-weight: 700;
    }
</style>
"""


def count_people() -> int:
    return len([p for p in DATASET_DIR.iterdir() if p.is_dir()]) if DATASET_DIR.exists() else 0


def count_images() -> int:
    if not DATASET_DIR.exists():
        return 0
    return sum(1 for path in DATASET_DIR.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def render_header() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
            <h1>Face Recognition Attendance System</h1>
            <p class="small-muted">Deep learning face embeddings, real-time webcam recognition, and duplicate-safe CSV attendance.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard() -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f"<div class='metric-box'><b>People</b><h2>{count_people()}</h2></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-box'><b>Images</b><h2>{count_images()}</h2></div>", unsafe_allow_html=True)
    col3.markdown(
        f"<div class='metric-box'><b>Model</b><h2>{'Ready' if CLASSIFIER_PATH.exists() else 'Missing'}</h2></div>",
        unsafe_allow_html=True,
    )
    col4.markdown(
        f"<div class='metric-box'><b>Attendance</b><h2>{'CSV' if ATTENDANCE_PATH.exists() else 'Empty'}</h2></div>",
        unsafe_allow_html=True,
    )


def add_person_view() -> None:
    st.subheader("Add New Person")
    st.write("Capture clear front-facing samples. Vary expression and head angle slightly for better recognition.")

    with st.form("capture_form"):
        person_name = st.text_input("Person name", placeholder="Example: Aisha Khan")
        image_count = st.slider("Images to capture", min_value=10, max_value=100, value=30, step=5)
        camera_index = st.number_input("Camera index", min_value=0, max_value=5, value=0, step=1)
        submitted = st.form_submit_button("Start Webcam Capture")

    if submitted:
        try:
            with st.spinner("Opening webcam. Press q in the camera window to stop early."):
                person_dir = capture_person_images(person_name, int(image_count), int(camera_index))
            st.success(f"Saved training images in {person_dir}")
        except Exception as exc:
            st.error(f"Capture failed: {exc}")

    st.divider()
    st.write("You can also upload images manually.")
    upload_name = st.text_input("Folder/person name for uploads", key="upload_name")
    uploads = st.file_uploader("Upload face images", type=["jpg", "jpeg", "png", "bmp", "webp"], accept_multiple_files=True)

    if st.button("Save Uploaded Images", disabled=not uploads):
        try:
            from src.face_system import clean_person_name

            safe_name = clean_person_name(upload_name)
            if not safe_name:
                st.error("Enter a valid person name before saving uploads.")
                return
            person_dir = DATASET_DIR / safe_name
            person_dir.mkdir(parents=True, exist_ok=True)
            for index, uploaded_file in enumerate(uploads, start=1):
                suffix = Path(uploaded_file.name).suffix.lower() or ".jpg"
                (person_dir / f"{safe_name}_upload_{index:03d}{suffix}").write_bytes(uploaded_file.getbuffer())
            st.success(f"Saved {len(uploads)} images to {person_dir}")
        except Exception as exc:
            st.error(f"Upload save failed: {exc}")


def train_view() -> None:
    st.subheader("Train Model")
    st.write("Training extracts DeepFace embeddings, then fits a classifier for names.")

    classifier_type = st.radio("Classifier", ["svm", "knn"], horizontal=True)
    model_name = st.selectbox(
        "DeepFace embedding model",
        ["Facenet", "VGG-Face", "ArcFace", "OpenFace", "DeepFace"],
        index=0,
        help="Facenet is a good default for accuracy and speed.",
    )

    if st.button("Train Model", type="primary"):
        try:
            with st.spinner("Extracting embeddings and training classifier..."):
                result = train_classifier(classifier_type=classifier_type, model_name=model_name)
            st.success(f"Model saved to {result['model_path']}")
            st.write(f"Classes: {', '.join(result['classes'])}")
            if result["accuracy"] is not None:
                st.write(f"Validation accuracy: {result['accuracy'] * 100:.2f}%")
            if result["skipped"]:
                with st.expander("Skipped images"):
                    st.write(result["skipped"])
        except Exception as exc:
            st.error(f"Training failed: {exc}")


def image_video_view() -> None:
    st.subheader("Recognize From Image or Video")
    try:
        model_payload = load_model()
    except Exception as exc:
        st.warning(str(exc))
        return

    tolerance = st.slider("Recognition confidence threshold", 0.1, 0.95, 0.55, 0.01)
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])

    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as temp_file:
            temp_file.write(uploaded.getbuffer())
            temp_path = temp_file.name

        frame = cv2.imread(temp_path)
        if frame is None:
            st.error("Could not read uploaded image.")
            return
        results = recognize_faces(frame, model_payload, tolerance=tolerance)
        annotated = draw_results(frame, results)
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        st.write([{"name": r.name, "confidence": round(r.confidence, 3)} for r in results])


def realtime_view() -> None:
    st.subheader("Real-Time Recognition")
    try:
        model_payload = load_model()
    except Exception as exc:
        st.warning(str(exc))
        return

    col1, col2, col3, col4 = st.columns(4)
    camera_index = col1.number_input("Camera index", min_value=0, max_value=5, value=0, step=1)
    tolerance = col2.slider("Confidence threshold", 0.1, 0.95, 0.55, 0.01)
    resize_scale = col3.slider("Processing scale", 0.25, 1.0, 0.5, 0.05, help="Lower values improve FPS on low-end PCs.")
    duration = col4.slider("Run time", 5, 300, 30, 5)
    mark_csv = st.checkbox("Mark attendance automatically", value=True)
    run = st.button("Start Recognition", type="primary")

    frame_placeholder = st.empty()
    status_placeholder = st.empty()

    if run:
        cap = cv2.VideoCapture(int(camera_index))
        if not cap.isOpened():
            st.error("Could not open webcam. Check camera permissions or camera index.")
            return

        previous_time = time.time()
        end_time = time.time() + duration
        try:
            while time.time() < end_time:
                ok, frame = cap.read()
                if not ok:
                    st.error("Could not read from webcam.")
                    break

                results = recognize_faces(frame, model_payload, tolerance=tolerance, resize_scale=resize_scale)
                if mark_csv:
                    for result in results:
                        mark_attendance(result.name)

                now = time.time()
                fps = 1.0 / max(now - previous_time, 1e-6)
                previous_time = now
                annotated = draw_results(frame, results, fps=fps)
                frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                status_placeholder.caption(f"Detected {len(results)} face(s). FPS: {fps:.1f}")
                time.sleep(0.01)
        finally:
            cap.release()


def attendance_view() -> None:
    st.subheader("Attendance")
    if not ATTENDANCE_PATH.exists():
        st.info("No attendance has been recorded yet.")
        return

    df = pd.read_csv(ATTENDANCE_PATH)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("Download CSV", ATTENDANCE_PATH.read_bytes(), file_name="attendance.csv", mime="text/csv")


def main() -> None:
    ensure_project_dirs()
    render_header()
    if DeepFace is None:
        st.error(DEEPFACE_INSTALL_HELP)
        st.code(
            "python -m venv .venv\n"
            ".venv\\Scripts\\activate\n"
            "python -m pip install --upgrade pip\n"
            "pip install deepface tf-keras\n"
            "pip install -r requirements.txt",
            language="powershell",
        )
        st.info(f"The app uses DeepFace embeddings. Default model: {DEFAULT_DEEPFACE_MODEL}.")
        return

    render_dashboard()

    page = st.sidebar.radio(
        "Menu",
        ["Add New Person", "Train Model", "Start Recognition", "Recognize Image", "View Attendance"],
    )

    if page == "Add New Person":
        add_person_view()
    elif page == "Train Model":
        train_view()
    elif page == "Start Recognition":
        realtime_view()
    elif page == "Recognize Image":
        image_video_view()
    else:
        attendance_view()


if __name__ == "__main__":
    main()
