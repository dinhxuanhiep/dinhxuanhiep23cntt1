import streamlit as st
from PIL import Image
import numpy as np
import cv2
from yolo import YOLO

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return YOLO(cuda=False)

yolo = load_model()

# --- GIAO DIỆN ---
st.set_page_config(page_title="Phát Hiện Biển Số", layout="wide")
st.title("🚗 Hệ Thống Phát Hiện Vị Trí Biển Số Xe")
st.markdown("*(Phiên bản YOLOv4-tiny - Chỉ khoanh vùng, không đọc số)*")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Ảnh đầu vào")
    uploaded_file = st.file_uploader("Chọn ảnh xe...", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Ảnh gốc', use_column_width=True)
        
        if st.button('🔍 Quét Ảnh', type="primary"):
            with st.spinner('AI đang tìm vị trí biển số...'):
                r_image, boxes, scores = yolo.detect_image(image)
                
                with col2:
                    st.subheader("2. Kết quả")
                    st.image(r_image, caption='Kết quả phát hiện', use_column_width=True)
                    
                    if len(boxes) > 0:
                        st.success(f"Đã phát hiện {len(boxes)} biển số!")
                        st.write("Chi tiết các vùng biển số:")
                        cols = st.columns(len(boxes))
                        for i, box in enumerate(boxes):
                            top, left, bottom, right = box
                            crop_img = image.crop((left, top, right, bottom))
                            with cols[i if i < len(cols) else 0]:
                                st.image(crop_img, width=150, caption=f"Độ tin cậy: {scores[i]:.2f}")
                    else:

                        st.warning("Không tìm thấy biển số nào.")
