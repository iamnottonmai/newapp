import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2
import os
import gdown

# 🟢 ต้องอยู่บรรทัดแรก
st.set_page_config(page_title="Scoliosis")

# โหลดโมเดล (โหลดแค่ครั้งเดียว)
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'best.pt')

    # 🔽 If model doesn't exist, download from Google Drive
    if not os.path.exists(model_path):
        file_id = "1HGdlajuTx8ly0zc-rmYMd0ni4kHIoTv-"  # ⬅️ Replace this
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)

    return YOLO(model_path)

model = load_model()

# ✅ CSS Styling
st.markdown("""
    <style>
        .stApp {
            background-color: #4a4c51;
            min-height: 100vh;
            color: white;
            padding: 40px;
        }
        h1 {
            color: white;
            font-weight: bold;
            font-size: 48px;
            text-align: center;
            margin-bottom: 40px;
        }
    </style>
""", unsafe_allow_html=True)

# ✅ Heading
st.markdown("<h1>Scoliosis Detection</h1>", unsafe_allow_html=True)
st.markdown("<h2>Upload or Take a Picture</h2>", unsafe_allow_html=True)

# 📤 Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# 🖼️ Example images (smaller, and below uploader)
if uploaded_file is None:
    st.markdown("#### 🔍 ตัวอย่างภาพ")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image("IMG_1435_JPG_jpg.rf.7bf2e18e950b4245a10bda6dcc05036f.jpg", caption="ภาพตัวอย่างที่ 1", width=200)

    with col2:
        st.image("IMG_1436_JPG_jpg.rf.b5bdcd6762cd0ce96b33f81720ca160f.jpg", caption="ภาพตัวอย่างที่ 2", width=200)

# 📸 Camera input
camera_image = st.camera_input("Take a picture")

# 🧠 Function to run model and return result image + label
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
    result_text = "ตรวจพบ Scoliosis!" if detected_scoliosis else "ไม่พบความผิดปกติ"
    return result_image, result_text

# 🖼️ Predict and display
if uploaded_file is not None:
    image_pil = Image.open(uploaded_file)
    with st.spinner("กำลังวิเคราะห์..."):
        result_image, result_text = predict_and_draw(image_pil)
    st.image(result_image, caption=result_text, use_column_width=True)
    st.success(result_text)

elif camera_image is not None:
    image_pil = Image.open(camera_image)
    with st.spinner("กำลังวิเคราะห์..."):
        result_image, result_text = predict_and_draw(image_pil)
    st.image(result_image, caption=result_text, use_column_width=True)
    st.success(result_text)
