import streamlit as st
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import albumentations as A
import numpy as np
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt
import pandas as pd
import altair as alt
import gdown

# --- 1. CONFIGURATION & STYLE ---
st.set_page_config(page_title="DeepLook Analytics", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    div[data-testid="stMetricValue"] { font-size: 28px; color: #00e676; font-family: 'Helvetica Neue'; }
    h1 { color: #ffffff; text-align: center; font-weight: 300; letter-spacing: 2px; }
    h3 { color: #00e676; font-weight: 400; }
    .stButton>button { width: 100%; background-color: #00e676; color: black; font-weight: bold; border-radius: 8px; border: none; padding: 10px; transition: 0.3s;}
    .stButton>button:hover { background-color: #00b359; color: white; }
</style>
""", unsafe_allow_html=True)

# --- 2. MODEL DEFINITION ---
class SiameseUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=2)
        old_layer = self.base_model.encoder.conv1
        new_layer = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_layer.weight[:, :3] = old_layer.weight
            new_layer.weight[:, 3:] = old_layer.weight
        self.base_model.encoder.conv1 = new_layer

    def forward(self, t1, t2):
        x = torch.cat([t1, t2], dim=1)
        return self.base_model(x)

# --- 3. SMART MODEL LOADER (Auto-Download from Drive) ---
@st.cache_resource
def load_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SiameseUNet().to(device)
    
    local_path = 'levir_cd_final_model.pth'
    
    # ---------------------------------------------------------
    # [ACTION REQUIRED] PASTE YOUR GOOGLE DRIVE FILE ID BELOW
    # ---------------------------------------------------------
    file_id = '1KV_BKJKYu4LQFScsFAfHbJJ-AYpt7n2y' 
    
    # Download if missing
    if not os.path.exists(local_path):
        url = f'https://drive.google.com/uc?id={file_id}'
        try:
            with st.spinner("📥 Downloading Model from Cloud (One-time setup)..."):
                gdown.download(url, local_path, quiet=False)
            st.success("✅ Model Downloaded Successfully!")
        except Exception as e:
            st.error(f"❌ Failed to download model: {e}")
            return None, device

    # Load Weights
    try:
        # If using the quantized/shrunk model, uncomment the next line:
        # model.half() 
        state_dict = torch.load(local_path, map_location=device)
        model.load_state_dict(state_dict)
        # If using quantized model, verify we inflate back to float for CPU inference:
        model.float() 
    except Exception as e:
        st.error(f"❌ Error loading weights: {e}")
        return None, device
    
    return model, device

model, device = load_model()

# --- 4. ALIGNMENT ENGINE (Fixes Shifted Images) ---
def align_images(img1, img2):
    # Convert PIL to BGR (OpenCV standard)
    img1_cv = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
    img2_cv = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
    
    gray1 = cv2.cvtColor(img1_cv, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_cv, cv2.COLOR_BGR2GRAY)
    
    # Detect Features
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    
    # Match Features
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    if des1 is None or des2 is None: return img2 # Safety check
    
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:int(len(matches) * 0.15)]
    
    if len(good_matches) < 4: return img2 # Not enough matches, return original
    
    # Extract Points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Warp
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    if M is None: return img2
        
    h, w, _ = img1_cv.shape
    aligned_img2 = cv2.warpPerspective(img2_cv, M, (w, h))
    
    # Return as PIL Image (RGB)
    return Image.fromarray(cv2.cvtColor(aligned_img2, cv2.COLOR_BGR2RGB))

# --- 5. UI & LOGIC ---
st.title("🛰️ DeepLook: Satellite Intelligence")
st.markdown("---")

allowed = ['png', 'jpg', 'jpeg', 'tif', 'tiff']

with st.container():
    col1, col2 = st.columns(2)
    f1 = col1.file_uploader("📂 Time 1 (Before)", type=allowed)
    f2 = col2.file_uploader("📂 Time 2 (After)", type=allowed)

if f1 and f2:
    image1 = Image.open(f1).convert("RGB")
    image2 = Image.open(f2).convert("RGB")
    
    # Auto-Align
    with st.spinner("🔄 Aligning Satellite Imagery..."):
        try: image2 = align_images(image1, image2)
        except: pass

    # Display Input
    c1, c2 = st.columns(2)
    c1.image(image1, caption="Before", use_container_width=True)
    c2.image(image2, caption="After (Aligned)", use_container_width=True)

    st.markdown("###")
    if st.button("🚀 RUN INTELLIGENCE ANALYSIS"):
        if model is None:
            st.error("Model not loaded. Check Google Drive ID.")
        else:
            with st.spinner("Running Neural Network..."):
                # Preprocess
                transform = A.Compose([A.Resize(256, 256)])
                img1_np, img2_np = np.array(image1), np.array(image2)
                aug = transform(image=img1_np, mask=img2_np)
                t1, t2 = aug['image'], aug['mask']
                
                t1_tensor = torch.from_numpy(t1).permute(2, 0, 1).float()/255.0
                t2_tensor = torch.from_numpy(t2).permute(2, 0, 1).float()/255.0
                t1_tensor = t1_tensor.unsqueeze(0).to(device)
                t2_tensor = t2_tensor.unsqueeze(0).to(device)
                
                # Inference
                model.eval()
                with torch.no_grad():
                    pred_raw = model(t1_tensor, t2_tensor)
                    prob_map = torch.softmax(pred_raw, dim=1)[:, 1, :, :].squeeze().cpu().numpy()
                    mask = torch.argmax(pred_raw, dim=1).squeeze().cpu().numpy().astype(np.uint8)
                
                # Metrics
                total_pixels = 256 * 256
                changed_pixels = np.sum(mask == 1)
                pct_change = (changed_pixels / total_pixels) * 100
                
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
                obj_count = num_labels - 1 
                areas = stats[1:, 4] if obj_count > 0 else []
                
                if pct_change < 1: severity = "Low"
                elif pct_change < 10: severity = "Medium"
                else: severity = "High"

                # --- DASHBOARD ---
                st.markdown("---")
                st.subheader("📊 Intelligence Report")
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Change Coverage", f"{pct_change:.2f}%")
                m2.metric("Objects Detected", f"{obj_count}")
                m3.metric("Avg Object Size", f"{int(np.mean(areas)) if len(areas) > 0 else 0} px")
                m4.metric("Severity", severity)
                
                st.markdown("---")
                
                g1, g2 = st.columns([1, 1])
                with g1:
                    st.markdown("**🔍 Object Size Distribution**")
                    if len(areas) > 0:
                        df_areas = pd.DataFrame(areas, columns=["Pixel Area"])
                        chart = alt.Chart(df_areas).mark_bar(color='#00e676').encode(
                            alt.X("Pixel Area", bin=True),
                            y='count()',
                            tooltip=['count()']
                        ).properties(height=300)
                        st.altair_chart(chart, use_container_width=True)
                    else:
                        st.info("No significant changes detected.")

                with g2:
                    st.markdown("**🔥 AI Confidence Heatmap**")
                    fig, ax = plt.subplots()
                    im = ax.imshow(prob_map, cmap='magma')
                    plt.axis('off')
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    st.pyplot(fig, transparent=True)

                st.markdown("---")
                st.subheader("👁️ Visual Inspection")
                
                opacity = st.slider("Result Opacity", 0.0, 1.0, 0.4)
                
                mask_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
                mask_rgb[mask == 1] = [255, 0, 0] # Red
                
                overlay = Image.fromarray(mask_rgb).convert("RGBA")
                background = Image.fromarray(t2).convert("RGBA")
                blended = Image.blend(background, overlay, alpha=opacity)
                
                st.image(blended, caption="Interactive Result Overlay", width=700)
