import streamlit as st
import torch
import cv2
import numpy as np
from torchvision import transforms
from utils import crop_two_breasts, extract_tumor_region
from models import HybridCNNGNN

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Breast Tumor Detection",
    page_icon="🧬",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.main {
    background-color:#f4f6fb;
}

h1 {
    text-align:center;
    color:#1f4e79;
}

.block-container{
    padding-top:2rem;
}

.stButton>button{
    background-color:#1f77b4;
    color:white;
    border-radius:10px;
    height:3em;
    width:220px;
    font-size:18px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- PARAMETERS ----------------
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = HybridCNNGNN(use_gnn=False, trustnet=True).to(DEVICE)
    model.load_state_dict(torch.load("cnn_tumor_contour_model.pth", map_location=DEVICE))
    model.eval()
    return model

model = load_model()

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# ---------------- SIDEBAR ----------------
with st.sidebar:

    st.title("🧬 Breast AI Detector")

    st.write("Upload a breast thermogram image to detect abnormal tumor regions.")

    st.info("""
    **Processing Pipeline**

    1️⃣ Image Upload  
    2️⃣ Breast Region Detection  
    3️⃣ Breast Curve Extraction  
    4️⃣ Tumor Region Extraction  
    5️⃣ CNN Prediction
    """)

    st.success("Model: Hybrid CNN + TrustNet")

# ---------------- TITLE ----------------
st.title("Breast Thermogram Tumor Detection System")

uploaded_file = st.file_uploader(
    "Upload Thermogram Image",
    type=["bmp", "jpg", "png"]
)

# ---------------- IMAGE PROCESSING ----------------
if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Step 1: Crop Breast Region
    cropped = crop_two_breasts(img)

    # Step 2: Extract Tumor Region
    tumor_img = extract_tumor_region(cropped)

    # ---------------- DISPLAY IMAGES ----------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original Image")
        st.image(img, use_container_width=True)

    with col2:
        st.subheader("Breast Region")
        st.image(cropped, use_container_width=True)

    with col3:
        st.subheader("Tumor Region")
        st.image(tumor_img, use_container_width=True)

    st.divider()

    # ---------------- PREDICTION ----------------
    if st.button("🔍 Run Tumor Detection"):

        try:

            img_tensor = transform(tumor_img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(img_tensor)
                prob = torch.sigmoid(output).item()

            st.subheader("Prediction Result")

            if prob > 0.5:
                st.error(f"⚠ Tumor Detected")
                st.write(f"Confidence: **{prob*100:.2f}%**")
            else:
                st.success("✅ Healthy Breast Tissue")
                st.write(f"Confidence: **{(1-prob)*100:.2f}%**")

            st.divider()

            # Download tumor image
            st.download_button(
                label="⬇ Download Tumor Region",
                data=cv2.imencode(".jpg", tumor_img)[1].tobytes(),
                file_name="tumor_region.jpg",
                mime="image/jpeg"
            )

        except Exception as e:

            st.error("Error during prediction.")
            st.write(e)
