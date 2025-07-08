import streamlit as st
from PIL import Image, ImageOps
from ultralytics import YOLO
import numpy as np
import cv2
import os
import gdown
from html import escape
import torch
import open_clip
from torchvision import transforms
import datetime

st.set_page_config(page_title="Scoliosis")

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'best.pt')
    if not os.path.exists(model_path):
        file_id = "1HGdlajuTx8ly0zc-rmYMd0ni4kHIoTv-"
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
    return YOLO(model_path)

model = load_model()

@st.cache_resource
def load_clip_model():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    return model.eval(), preprocess, tokenizer

clip_model, clip_preprocess, clip_tokenizer = load_clip_model()

def is_image_human_back(image_pil):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess, tokenizer = clip_model.to(device), clip_preprocess, clip_tokenizer

    image = preprocess(image_pil).unsqueeze(0).to(device)

    texts = [
        "a photo of a bare human back",
        "a photo of a hand",
        "a wall",
        "a photo of a face",
        "a random object",
        "a tree",
        "a close-up",
        "a foot"
    ]
    text_tokens = tokenizer(texts).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        top_idx = similarity.argmax().item()
        top_label = texts[top_idx]
        top_score = similarity[0, top_idx].item()

    return top_label == "a photo of a bare human back" and top_score > 0.25

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

        .stFileUploader, .blue-box {
            background-color: #e6f0ff !important;
            border: 2px dashed #4a90e2 !important;
            padding: 20px !important;
            border-radius: 10px;
            margin-bottom: 20px;
        }

        .stFileUploader div:first-child {
            color: black !important;
            font-weight: bold;
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
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>Scoliosis Detection</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='color:black;'>อัปโหลด ถ่ายภาพ หรือเลือกภาพตัวอย่าง</h2>", unsafe_allow_html=True)

col_upload, col_test = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader("อัปโหลดภาพ", type=["jpg", "jpeg", "png"])

with col_test:
    st.markdown('<div class="blue-box"><b>เลือกภาพตัวอย่าง</b>', unsafe_allow_html=True)
    test_image_folder = "test_images"
    test_image_files = sorted([f for f in os.listdir(test_image_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))])

    selected_test_image = st.selectbox(
        "เลือกจากภาพตัวอย่าง",
        [""] + test_image_files,
        format_func=lambda x: "เลือกภาพ" if x == "" else x,
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if selected_test_image:
        st.image(os.path.join(test_image_folder, selected_test_image), width=250, caption="ภาพตัวอย่างที่เลือก")

if not selected_test_image:
    st.markdown("""
    <div style='margin-top: -10px; margin-bottom: 20px; color: black; font-weight: bold;'>
    <u>คำแนะนำในการถ่ายภาพ</u>:
    <ol style="margin-top:10px;">
        <li><b>เสื้อผ้า:</b> ให้แน่ใจว่าหลังเปลือย ไม่มีเสื้อผ้า ผม หรือเครื่องประดับบดบังแนวกระดูกสันหลัง</li>
        <li><b>ระยะห่าง:</b> ตั้งกล้องให้ไกลพอที่จะถ่ายได้ทั้งหลัง ตั้งแต่ไหล่ถึงสะโพก</li>
        <li><b>ท่าทาง:</b> ยืนตรง หันหลังให้กล้อง วางแขนข้างลำตัวตามธรรมชาติ</li>
        <li><b>แสงสว่าง:</b> ใช้แสงที่สว่างสม่ำเสมอ หลีกเลี่ยงเงาหรือแสงย้อน</li>
        <li><b>มุมกล้อง:</b> ตั้งกล้องให้สูงประมาณไหล่ ไม่เอียงขึ้นหรือลง</li>
        <li><b>พื้นหลัง:</b> ใช้พื้นหลังเรียบ สีอ่อน เช่น ผนังขาว</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)


camera_image = None
if uploaded_file is None and not selected_test_image:
    camera_image = st.camera_input("ถ่ายภาพ")

show_example_images = uploaded_file is None and not selected_test_image
if show_example_images:
    with st.expander("📸 คลิกเพื่อดูภาพตัวอย่าง"):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image("IMG_1436_JPG_jpg.rf.b5bdcd6762cd0ce96b33f81720ca160f.jpg", width=250)
        with col2:
            st.image("IMG_1435_JPG_jpg.rf.7bf2e18e950b4245a10bda6dcc05036f.jpg", width=250)

def predict_and_draw(image_pil):
    confidence_threshold = 0.4
    image = np.array(image_pil)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    results = model(image)
    result = results[0]
    boxes = result.boxes if result.boxes is not None else []
    detected_scoliosis = False

    for box in boxes:
        conf = float(box.conf[0])
        if conf < confidence_threshold:
            continue

        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if label == "scoliosis":
            detected_scoliosis = True
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    result_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    result_text = "ตรวจพบ Scoliosis ควรได้รับการประเมินเพิ่มเติมจากแพทย์" if detected_scoliosis else "ไม่พบความผิดปกติ"
    return result_image, result_text

def display_results(image_pil):
    with st.spinner("กำลังตรวจสอบภาพ..."):
        if not is_image_human_back(image_pil):
            st.markdown(f"""
                <div style=\"background-color:#fff3cd; padding: 10px; border-radius: 5px; color: #856404; font-weight: bold; text-align:center;\">
                    ❌ ภาพนี้ไม่ใช่ภาพด้านหลังของมนุษย์ที่เปลือย กรุณาอัปโหลดภาพที่เหมาะสม
                </div>
            """, unsafe_allow_html=True)
            return

    with st.spinner("กำลังวิเคราะห์..."):
        result_image, result_text = predict_and_draw(image_pil)

    st.image(result_image, use_container_width=True)
    escaped_text = escape(result_text)

    if "Scoliosis" in result_text:
        st.markdown(f"""
            <div style=\"background-color:#ffcccc; padding: 10px; border-radius: 5px; color: black; font-weight: bold; text-align:center;\">
                {escaped_text}
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style=\"background-color:#c8e6c9; padding: 10px; border-radius: 5px; color: black; font-weight: bold; text-align:center;\">
                {escaped_text}
            </div>
        """, unsafe_allow_html=True)

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file)
    display_results(image_pil)
elif camera_image is not None:
    image_pil = ImageOps.mirror(Image.open(camera_image))
    display_results(image_pil)
elif selected_test_image:
    image_pil = Image.open(os.path.join(test_image_folder, selected_test_image))
    display_results(image_pil)

startup_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

st.markdown(
    f"""
    <div style='position: fixed; bottom: 10px; left: 15px; color: gray; font-size: 0.85em; z-index: 9999;'>
        แอปเริ่มต้นล่าสุดเมื่อ: {startup_time}
    </div>
    """,
    unsafe_allow_html=True
)
