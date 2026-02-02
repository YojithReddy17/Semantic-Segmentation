import streamlit as st
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import albumentations as A
import numpy as np
import cv2
from PIL import Image
import os
import pandas as pd
import gdown
from streamlit_image_comparison import image_comparison

# --- 1. CONFIGURATION & STYLE ---
st.set_page_config(page_title="DeepLook Pro", layout="wide", page_icon="🛰️")

# Enterprise Dark Theme CSS
st.markdown("""
<style>
    .main { background-color: #0f1116; }
    h1 { font-family: 'Helvetica Neue', sans-serif; font-weight: 300; letter-spacing: 2px; color: #fff; }
    .stMetric { background-color: #1e212b; padding: 15px; border-radius: 10px; border: 1px solid #333; }
    .stButton>button { 
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%); 
        color: black; 
        font-weight: bold; 
        border: none; 
        height: 50px; 
        font-size: 18px;
        width: 100%;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(0,201,255,0.4);
    }
</style>
""", unsafe_allow_html=True)

# --- 2. MODEL DEFINITION ---
class SiameseUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # We use a shared encoder (ResNet18)
        self.base_model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=2)
        
        # Modify first layer to handle Siamese input if needed (Conceptually)
        # However, for this implementation, we use the standard trick of modifying weights
        # to accept concatenated input or shared processing.
        # *Note:* To match your training code exactly, we ensure the first layer matches 
        # whatever structure you saved. If you used standard 6-channel input:
        old_layer = self.base_model.encoder.conv1
        new_layer = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_layer.weight[:, :3] = old_layer.weight
            new_layer.weight[:, 3:] = old_layer.weight
        self.base_model.encoder.conv1 = new_layer

    def forward(self, t1, t2):
        x = torch.cat([t1, t2], dim=1)
        return self.base_model(x)

# --- 3. SMART MODEL LOADER ---
@st.cache_resource
def load_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SiameseUNet().to(device)
    
    local_path = 'levir_cd_final_model.pth'
    
    # ==========================================
    # [ACTION REQUIRED] PASTE DRIVE ID BELOW
    # ==========================================
    file_id = '1KV_BKJKYu4LQFScsFAfHbJJ-AYpt7n2y' 
    
    # Auto-Download from Google Drive if missing
    if not os.path.exists(local_path):
        url = f'https://drive.google.com/uc?id={file_id}'
        try:
            with st.spinner("📥 Downloading AI Model from Cloud... (One-time setup)"):
                gdown.download(url, local_path, quiet=False)
            st.success("✅ Model Downloaded Successfully!")
        except Exception as e:
            st.error(f"❌ Failed to download model. Check File ID. Error: {e}")
            return None, device

    # Load Weights
    try:
        # If you saved the quantized model, uncomment: model.half()
        state_dict = torch.load(local_path, map_location=device)
        model.load_state_dict(state_dict)
        model.float() # Ensure float32 for CPU compatibility
    except Exception as e:
        st.error(f"❌ Error loading weights: {e}")
        return None, device
    
    return model, device

model, device = load_model()

# --- 4. ALIGNMENT ENGINE ---
def align_images(img1, img2):
    # Convert to OpenCV format
    img1_cv = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
    img2_cv = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
    
    gray1 = cv2.cvtColor(img1_cv, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_cv, cv2.COLOR_BGR2GRAY)
    
    # Feature Matching
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    if des1 is None or des2 is None: return img2
    
    matches = sorted(matcher.match(des1, des2), key=lambda x: x.distance)
    good = matches[:int(len(matches) * 0.15)]
    
    if len(good) < 4: return img2
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    if M is None: return img2
    
    h, w, _ = img1_cv.shape
    aligned = cv2.warpPerspective(img2_cv, M, (w, h))
    return Image.fromarray(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))

# --- 5. UI LAYOUT ---
st.title("🛰️ DeepLook Pro")
st.markdown("### Satellite Construction Intelligence System")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Calibration")
    resolution = st.number_input("Pixel Resolution (meters/px)", value=0.5, step=0.1, help="LEVIR-CD is 0.5m/px.")
    st.info(f"At {resolution}m/px, a 256x256 image covers {(256*resolution)**2/10000:.2f} hectares.")
    
    st.markdown("---")
    st.markdown("**System Status**")
    if device == 'cuda': st.success("🟢 GPU Online")
    else: st.warning("🟡 CPU Mode (Slower)")

# --- UPLOAD SECTION (Independent Display) ---
c1, c2 = st.columns(2)

with c1:
    f1 = st.file_uploader("Time 1 (Before)", type=['jpg','png','tif'])
    if f1:
        st.image(f1, caption="Time 1 Uploaded", use_container_width=True)

with c2:
    f2 = st.file_uploader("Time 2 (After)", type=['jpg','png','tif'])
    if f2:
        st.image(f2, caption="Time 2 Uploaded", use_container_width=True)

# --- EXECUTION LOGIC ---
if f1 and f2:
    img1 = Image.open(f1).convert("RGB")
    img2 = Image.open(f2).convert("RGB")
    
    st.markdown("---")
    
    if st.button("🚀 RUN INTELLIGENCE ANALYSIS", type="primary"):
        if model is None:
            st.error("Model failed to load. Check your Google Drive ID.")
        else:
            with st.spinner("🤖 Aligning Images & Detecting Changes..."):
                # 1. Alignment
                try: img2 = align_images(img1, img2)
                except: pass
                
                # 2. Preprocessing
                transform = A.Compose([A.Resize(256, 256)])
                aug = transform(image=np.array(img1), mask=np.array(img2))
                t1_raw, t2_raw = aug['image'], aug['mask']
                
                t1 = torch.from_numpy(t1_raw).permute(2,0,1).float().unsqueeze(0).to(device)/255.0
                t2 = torch.from_numpy(t2_raw).permute(2,0,1).float().unsqueeze(0).to(device)/255.0
                
                # 3. Inference
                model.eval()
                with torch.no_grad():
                    pred = model(t1, t2)
                    mask = torch.argmax(pred, dim=1).squeeze().cpu().numpy().astype(np.uint8)

            # --- VISUAL RESULTS ---
            st.markdown("### 👁️ Visual Change Inspection")
            
            # Create Overlay for Slider (Red tint on new buildings)
            mask_colored = np.zeros_like(t2_raw)
            mask_colored[mask == 1] = [255, 0, 0]
            result_overlay = cv2.addWeighted(t2_raw, 0.7, mask_colored, 0.3, 0)
            
            # Slider Component
            image_comparison(
                img1=t1_raw,
                img2=result_overlay,
                label1="Before",
                label2="After + AI Detection",
                width=700,
                starting_position=50,
                show_labels=True,
                make_responsive=True,
                in_memory=True
            )

            # --- ANALYTICS DASHBOARD ---
            st.markdown("---")
            st.markdown("### 📊 Construction Analysis Report")
            
            pixel_count = np.sum(mask == 1)
            real_area_m2 = pixel_count * (resolution ** 2)
            coverage = (pixel_count / (256*256)) * 100
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Changed Pixels", f"{pixel_count:,}")
            col_m2.metric("Est. Area (m²)", f"{real_area_m2:,.2f}")
            col_m3.metric("Est. Area (Acres)", f"{real_area_m2 * 0.000247105:,.3f}")
            col_m4.metric("Change Coverage", f"{coverage:.2f}%")
            
            # Download Report
            st.markdown("###")
            report = f"""
            DEEPLOOK PRO - AUDIT REPORT
            ===========================
            Resolution used: {resolution} m/px
            Total Detected Change: {real_area_m2:.2f} sq meters
            Coverage Percentage: {coverage:.2f}%
            """
            st.download_button("📄 Download Text Report", report, file_name="change_audit.txt")
