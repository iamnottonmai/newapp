import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2
import os
import gdown

# 🟢 Page config
st.set_page_config(page_title="Scoliosis Detection", layout="centered")

# 📥 Load model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'best.pt')

    # 🔽 Download if missing
    if not os.path.exists(model_path):
        file_id = "1HGdlajuTx8ly0zc-rmYMd0ni4kHIoTv-"
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)

    return YOLO(model_path)

model = load_model()

# 🎨 Style
st.markdown("""
    <style>
        .stApp {
            background-color: #3470ac;
            min-height: 100vh;
            color: white;
            padding: 40px;
        }
        h1, h2 {
            text-align: center;
            font-weight: bold;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# 🏷️ Title
st.markdown("<h1>Scoliosis Detection</h1>", unsafe_allow_html=True)
st.markdown("<h2>Upload or Take a Picture</h2>", unsafe_allow_html=True)

# 📤 Upload or camera
uploaded_file = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])
camera_image = st.camera_input("📸 Or take a picture")

# 🧠 Inference
def predict_and_draw(image_pil):
    image = np.array(image_pil)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    results = model(image)
    result = results[0]
    boxes = result.boxes if result.boxes is not None else []

    detected = False

    for box in boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if label == "scoliosis":
            detected = True
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    result_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    result_text = "✅ ตรวจพบ Scoliosis!" if detected else "✅ ไม่พบความผิดปกติ"
    return result_img, result_text

# ▶️ Run prediction
input_image = uploaded_file or camera_image

if input_image is not None:
    img = Image.open(input_image)
    with st.spinner("🔍 Analyzing..."):
        result_image, result_text = predict_and_draw(img)
    st.image(result_image, caption=result_text, use_column_width=True)
    st.success(result_text)
