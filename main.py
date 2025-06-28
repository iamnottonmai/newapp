import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2
import os
import gdown
from html import escape
import torch
import open_clip
from torchvision import transforms

st.set_page_config(page_title="Scoliosis")

# ✅ Load YOLO model
@st.cache_resource
def load_yolo_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'best.pt')
    if not os.path.exists(model_path):
        file_id = "1HGdlajuTx8ly0zc-rmYMd0ni4kHIoTv-"
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
    return YOLO(model_path)

model = load_yolo_model()

# ✅ Load CLIP model
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

# ✅ Prediction

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
    result_text = "Scoliosis detected. Further evaluation and treatment may be needed." if detected_scoliosis else "No abnormalities detected"
    return result_image, result_text

# ✅ Display results

def display_results(image_pil):
    with st.spinner("Checking image..."):
        if not is_image_human_back(image_pil):
            st.markdown(f"""
                <div style=\"background-color:#fff3cd; padding: 10px; border-radius: 5px; color: #856404; font-weight: bold; text-align:center;\">
                    ❌ This image does not appear to be a bare human back. Please upload a proper back photo.
                </div>
            """, unsafe_allow_html=True)
            return

    with st.spinner("Analysing..."):
        result_image, result_text = predict_and_draw(image_pil)

    st.image(result_image, use_container_width=True)
    escaped_text = escape(result_text)

    if "Scoliosis detected" in result_text:
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

# ✅ Image uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
camera_image = None
if uploaded_file is None:
    camera_image = st.camera_input("Take a picture")

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file)
    display_results(image_pil)
elif camera_image is not None:
    image_pil = Image.open(camera_image)
    display_results(image_pil)
