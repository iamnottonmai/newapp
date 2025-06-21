import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2
import os
import gdown
from html import escape

# 🟢 ต้องอยู่บรรทัดแรก
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

        h1 {
            color: black;
            font-weight: bold;
            font-size: 48px;
            text-align: left;
            margin-bottom: 40px;
        }

        .input-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            margin-bottom: 0px;
        }

        .input-box {
            background-color: #e6f0ff;
            border: 2px dashed #4a90e2;
            padding: 20px;
            border-radius: 10px;
        }

        .sample-caption {
            font-size: 14px;
            text-align: center;
            color: #444;
        }

        [data-testid="stCameraInput"] button {
            background-color: #e6f0ff !important;
            color: black !important;
            font-weight: bold;
        }

        label[for^="camera-input"] {
            color: white !important;
            font-weight: bold;
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)

# ✅ Heading
st.markdown("<h1>Scoliosis Detection</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='color:black;'>Upload, Take, or Use a Sample Image</h2>", unsafe_allow_html=True)

# 📤 Custom Grid Layout for Inputs
st.markdown('<div class="input-grid">', unsafe_allow_html=True)

# Upload image box
with st.container():
    st.markdown('<div class="input-box">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    st.markdown('</div>', unsafe_allow_html=True)

# Sample image box
with st.container():
    st.markdown('<div class="input-box">', unsafe_allow_html=True)
    st.markdown("**Use sample image**")

    test_image_folder = "test_images"
    test_image_files = sorted([f for f in os.listdir(test_image_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))])

    selected_test_image = st.selectbox(
        "Select from sample images",
        [""] + test_image_files,
        format_func=lambda x: "Select an image" if x == "" else x,
        label_visibility="collapsed"
    )

    if selected_test_image:
        st.image(os.path.join(test_image_folder, selected_test_image), width=250)
        st.markdown("<div class='sample-caption'>Selected sample image</div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# 📌 Submission instructions
st.markdown("""
<div style='margin-top: 5px; margin-bottom: 20px; color: black; font-weight: bold;'>
Photograph Submission Instructions:
<ol>
<li>Nothing should obstruct the back.</li>
<li>Stand far enough from the camera to see the entire back.</li>
</ol>
</div>
""", unsafe_allow_html=True)

# 📂 Optional: View example images
with st.expander("📸 Click to view example images"):
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image("IMG_1436_JPG_jpg.rf.b5bdcd6762cd0ce96b33f81720ca160f.jpg", width=250)
    with col2:
        st.image("IMG_1435_JPG_jpg.rf.7bf2e18e950b4245a10bda6dcc05036f.jpg", width=250)

# 📸 Camera input — Only show if nothing else is selected
camera_image = None
if uploaded_file is None and not selected_test_image:
    camera_image = st.camera_input("Take a picture")

# 🧠 Prediction
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

# 📊 Display results
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

# 🚀 Run model
if uploaded_file is not None:
    image_pil = Image.open(uploaded_file)
    display_results(image_pil)
elif camera_image is not None:
    image_pil = Image.open(camera_image)
    display_results(image_pil)
elif selected_test_image:
    image_pil = Image.open(os.path.join(test_image_folder, selected_test_image))
    display_results(image_pil)
