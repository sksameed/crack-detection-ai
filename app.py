import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="Crack Detection AI", layout="wide")

# ==============================
# LOAD MODEL (FAST)
# ==============================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("crack_pretrained_model.keras", compile=False)

model = load_model()
IMG_SIZE = 256

# ==============================
# PREDICTION
# ==============================
def predict(image):
    img = np.array(image)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_processed = preprocess_input(img_resized.astype(np.float32))

    pred = model.predict(np.expand_dims(img_processed, axis=0))[0]

    mask = (pred > 0.3).astype(np.float32)
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    mask_display = (mask * 255).astype(np.uint8)

    overlay = img_rgb.copy()
    overlay[mask > 0.5] = [255, 0, 0]

    crack_percentage = mask.mean() * 100

    if crack_percentage < 1:
        severity = "LOW"
    elif crack_percentage < 5:
        severity = "MEDIUM"
    else:
        severity = "HIGH"

    return img_rgb, mask_display, overlay, crack_percentage, severity


# ==============================
# 🎨 STYLE
# ==============================
st.markdown("""
<style>
.main-title {
    font-size: 36px;
    font-weight: bold;
}
.metric-box {
    padding: 15px;
    border-radius: 10px;
    background-color: #1f2a40;
    text-align: center;
    font-size: 20px;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# SIDEBAR (🔥 PROFESSIONAL TOUCH)
# ==============================
st.sidebar.title("⚙️ Settings")
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg","jpeg","png"])
analyze = st.sidebar.button("🔍 Analyze")

st.sidebar.markdown("---")
st.sidebar.info("Crack Detection using Deep Learning")

# ==============================
# MAIN TITLE
# ==============================
st.markdown('<p class="main-title">🧱 Crack Detection Dashboard</p>', unsafe_allow_html=True)
st.markdown("Analyze structural cracks with AI")

st.divider()

# ==============================
# MAIN CONTENT
# ==============================
if uploaded_file:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([2,1])

    # LEFT: IMAGE
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # RIGHT: RESULTS PANEL
    with col2:
        if analyze:
            with st.spinner("Analyzing..."):
                original, mask, overlay, percent, severity = predict(image)

            st.success("Analysis Done")

            st.markdown(f"""
            <div class="metric-box">
            Crack: <b>{percent:.2f}%</b>
            </div>
            """, unsafe_allow_html=True)

            if severity == "LOW":
                st.success(f"Severity: {severity}")
            elif severity == "MEDIUM":
                st.warning(f"Severity: {severity}")
            else:
                st.error(f"Severity: {severity}")

    st.divider()

    # RESULTS ROW
    if analyze:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(original, caption="Original", use_container_width=True)

        with col2:
            st.image(mask, caption="Mask", use_container_width=True)

        with col3:
            st.image(overlay, caption="Overlay", use_container_width=True)