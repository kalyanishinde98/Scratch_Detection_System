
from pathlib import Path

import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Application settings
st.set_page_config(
    page_title="Scratch Detection System",
    page_icon="🔍",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best.pt"

st.title("Smart Surface Scratch Detection")
st.write("Upload a steel surface image to detect scratches.")

# Check model
if not MODEL_PATH.exists():
    st.error("Model not found. Please place best.pt inside the models folder.")
    st.stop()

# Load model once
@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))

model = load_model()

# Upload image
uploaded_file = st.file_uploader(
    "Choose a steel surface image",
    type=["jpg", "jpeg", "png"]
)

confidence = st.slider(
    "Detection confidence",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(image, width=400)

    if st.button("Detect Scratches", type="primary"):

        with st.spinner("Detecting scratches..."):

            results = model.predict(
                source=image,
                conf=confidence,
                imgsz=320,
                device="cpu",
                verbose=False
            )

            result = results[0]

            annotated_image = result.plot(line_width=1)
            annotated_image = Image.fromarray(
                annotated_image[:, :, ::-1]
            )

            scratch_count = len(result.boxes)

        st.subheader("Detection Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Detected Scratches", scratch_count)

        with col2:
            st.metric("Confidence Threshold", f"{confidence:.0%}")

        st.image(
            annotated_image,
            caption="Scratch Detection Output",
            width=400
        )

        if scratch_count > 0:
            st.success(f"{scratch_count} possible scratch detection(s) found.")
        else:
            st.info("No scratches detected at the selected confidence threshold.")
