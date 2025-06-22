import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2
import os
import gdown
from html import escape

st.set_page_config(page_title="Scoliosis")

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'best.pt')
    if not os.path.exists(model_path):
        file_id = "1HGdlajuTx8ly0zc-rmYMd0ni4kHIoTv-"
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
    return YOLO(model_path)

model = load_model()

# ✅ CSS Styling
st.markdown("""
    <style>
        .stApp {
            background-color: white;
            color: black;
            padding: 40px;
        }

        .stFileUploader {
            background-color: #e6f0ff !important;
            border: 2px dashed #4a90e2 !important;
            padding: 20px !important;
            border-radius: 10px;
        }

        .stFileUploader div:first-child {
            color: white !important;
            font-weight: bold;
        }

        .stFileUploader label {
            color: black !important;
        }

        label[for^="camera-input"] {
            color: white !important;
            font-weight: bold;
            font-size: 18px;
        }

        [data-testid="stCameraInput"] button {
            background-color: #e6f0ff !important;
            color: black !important;
            font-weight: bold;
        }

        details summary {
            font-weight: bold;
            font-size: 16px;
        }

        .retake-button {
            background-color: #4a90e2;
            color: white;
            padding: 8px 16px;
            border-radius: 8px;
            text-align: center;
            display: inline-block;
            margin-top: 10px;
            cursor: pointer;
        }
    </style>
""", unsafe_allow_html=True)

# ✅ Heading with logo (not clickable)
st.markdown("""
<div style="display: flex; align-items: center; gap: 20px; margin-bottom: 40px;">
    <img src="logo.png" alt="Logo" style="height: 60px;">
    <h1 style="color: black; font-size: 48px; margin: 0;">Scoliosis Detection</h1>
</div>
""", unsafe_allow_html=True)

st.markdown("<h2 style='color:black;'>Upload or Take a Picture</h2>", unsafe_allow_html=True)

# 📁 Test images
test_image_files = [f"test{i}.jpg" for i in range(1, 11)]
test_image_labels = [f"Test Image {i}" for i in range(1, 11)]
test_image_dict = dict(zip(test_image_labels, test_image_files))

# Upload & test selection side by side
col_upload, col_test = st.columns([3, 2])
with col_upload:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
with col_test:
    with st.expander("Choose sample photo"):
        selected_test = st.selectbox("Select a test image", test_image_labels)
        if selected_test:
            image_pil = Image.open(test_image_dict[selected_test])
            st.info("Test image selected. Running scoliosis detection...")

# 📌 Submission instructions
st.markdown("""
<div style='margin-top: 20px; margin-bottom: 20px; color: black; font-weight: bold;'>
Photograph Submission Instructions:
<ol>
<li>Nothing should obstruct the back.</li>
<li>Stand far enough from the camera to see the entire back.</li>
</ol>
</div>
""", unsafe_allow_html=True)

# 📂 Example images (dropdown)
with st.expander("📸 Click to view example images"):
    col1, col2 = st.columns(2)
    with col1:
        st.image("IMG_1436_JPG_jpg.rf.b5bdcd6762cd0ce96b33f81720ca160f.jpg", width=250)
    with col2:
        st.image("IMG_1435_JPG_jpg.rf.7bf2e18e950b4245a10bda6dcc05036f.jpg", width=250)

# 📸 Camera input
camera_image = st.camera_input("Take a picture")

# ✅ Prediction logic
def predict_and_draw(image_pil):
    image = np.array(image_pil)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    results = model(image)
    result = results[0]
    boxes = result.boxes if result.boxes is not None else []
    detected_scoliosis = False

    for box in boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if label == "scoliosis":
            detected_scoliosis = True
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    result_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    result_text = "Scoliosis detected. Further evaluation and treatment may be needed." if detected_scoliosis else "No abnormalities detected"
    return result_image, result_text

# ✅ Result display
def display_results(image_pil):
    with st.spinner("Analysing..."):
        result_image, result_text = predict_and_draw(image_pil)
    st.image(result_image, use_container_width=True)
    escaped_text = escape(result_text)
    if "Scoliosis detected" in result_text:
        st.markdown(f"""
            <div style="background-color:#ffcccc; padding: 10px; border-radius: 5px; color: black; font-weight: bold; text-align:center;">
                {escaped_text}
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style="background-color:#c8e6c9; padding: 10px; border-radius: 5px; color: black; font-weight: bold; text-align:center;">
                {escaped_text}
            </div>
        """, unsafe_allow_html=True)

# 🚀 Trigger prediction
if uploaded_file is not None:
    image_pil = Image.open(uploaded_file)
    display_results(image_pil)
elif camera_image is not None:
    image_pil = Image.open(camera_image)
    display_results(image_pil)
    st.markdown('<div class="retake-button">⬅️ Retake Photo (Reload Page)</div>', unsafe_allow_html=True)
elif selected_test:
    image_pil = Image.open(test_image_dict[selected_test])
    display_results(image_pil)
