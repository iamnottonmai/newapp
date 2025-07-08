import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import torch
import open_clip
from torchvision import transforms
import os
import gdown
from html import escape
import datetime
from ultralytics import YOLO
from torch.serialization import add_safe_globals

st.set_page_config(page_title="Scoliosis")

# ✅ Apply Noto Sans Thai font globally
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Thai&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Noto Sans Thai', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = "best.pt"
    
    # Download model from Google Drive if it doesn't exist
    if not os.path.exists(model_path):
        file_id = "1HGdlajuTx8ly0zc-rmYMd0ni4kHIoTv-"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)

    # Allow PyTorch to safely load YOLOv8 classification model
    # add_safe_globals({'ultralytics.models.yolo.classify.ClassificationModel': ClassificationModel})

    # Load model with map_location and safe globals
    model = torch.load(model_path, map_location=torch.device("cpu"))
    model.eval()
    return model

# Initialize the model
model = load_model()

# ✅ Load CLIP model for back validation
@st.cache_resource
def load_clip_model():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    return model.eval(), preprocess, tokenizer

clip_model, clip_preprocess, clip_tokenizer = load_clip_model()

# ✅ Classification label mapping
class_names = ["normal", "scoliosis"]

# ✅ Transform for classifier
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # adjust if RGB vs grayscale
])

# ✅ Verify image is a bare human back
def is_image_human_back(image_pil):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess, tokenizer = clip_model.to(device), clip_preprocess, clip_tokenizer

    image = preprocess(image_pil).unsqueeze(0).to(device)
    texts = [
        "a photo of a bare human back", "a photo of a hand", "a wall",
        "a photo of a face", "a random object", "a tree", "a close-up", "a foot"
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

col_upload, col_test = st.columns([1.5, 1])

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
        <li><b>เสื้อผ้า:</b> ไม่มีเสื้อผ้าบังแผ่นหลัง ไม่มีผมหรือเครื่องประดับที่บังแนวกระดูกสันหลัง</li>
        <li><b>ระยะห่าง:</b> ตั้งกล้องให้ไกลพอที่จะถ่ายได้ทั้งหลัง ตั้งแต่ไหล่ถึงสะโพก</li>
        <li><b>ท่าทาง:</b> ยืนตัวตรง หันหลังไปยังกล้อง และวางแขนไว้ที่ข้างลำตัวตามธรรมชาติ</li>
        <li><b>แสงสว่าง:</b> ใช้แสงสมดุล ไม่สว่างเกิน ไม่มืดหมองหรือมีเงา</li>
        <li><b>มุมกล้อง:</b> ความสูงของกล้องขนานกับแผ่นหลัง ไม่เอียงไปด้านบนหรือด้านล่าง</li>
        <li><b>พื้นหลัง:</b> ใช้พื้นหลังที่เรียบ สีอ่อน ไม่ฉูดฉาด</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

camera_image = None
if uploaded_file is None and not selected_test_image:
    camera_image = st.camera_input("ถ่ายภาพ")

show_example_images = uploaded_file is None and not selected_test_image
if show_example_images:
    with st.expander("คลิกเพื่อดูภาพตัวอย่าง"):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image("IMG_1436_JPG_jpg.rf.b5bdcd6762cd0ce96b33f81720ca160f.jpg", width=250)
        with col2:
            st.image("IMG_1435_JPG_jpg.rf.7bf2e18e950b4245a10bda6dcc05036f.jpg", width=250)

def classify_image(image_pil):
    image_tensor = transform(image_pil.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)
        label = class_names[pred.item()]
        return label, confidence.item()

def display_results(image_pil):
    with st.spinner("กำลังตรวจสอบภาพ..."):
        if not is_image_human_back(image_pil):
            st.markdown("""
                <div style="background-color:#fff3cd; padding: 10px; border-radius: 5px; color: #856404; font-weight: bold; text-align:center;">
                    ❌ ภาพนี้ไม่ใช่ภาพแผ่นหลังของมนุษย์ กรุณาอัปโหลดภาพที่เหมาะสม
                </div>
            """, unsafe_allow_html=True)
            return

    with st.spinner("กำลังวิเคราะห์..."):
        label, conf = classify_image(image_pil)

    st.image(image_pil, use_container_width=True)
    msg = f"ตรวจพบ Scoliosis ({conf*100:.1f}%)" if label == "scoliosis" else f"ไม่พบความผิดปกติ ({conf*100:.1f}%)"
    escaped_text = escape(msg)

    st.markdown(f"""
        <div style="background-color:{'#ffcccc' if label=='scoliosis' else '#c8e6c9'}; 
                    padding: 10px; border-radius: 5px; color: black; font-weight: bold; text-align:center;">
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

st.markdown(f"""
    <div style='position: fixed; bottom: 10px; left: 15px; color: gray; font-size: 0.85em; z-index: 9999;'>
        แอปเริ่มต้นล่าสุดเมื่อ: {startup_time}
    </div>
""", unsafe_allow_html=True)
