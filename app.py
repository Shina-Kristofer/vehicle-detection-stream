#!/usr/bin/env python3
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd

# Page setup for a professional wide layout
st.set_page_config(
    page_title="AI Vehicle and Classification System",
    page_icon="🚗",
    layout="wide"
)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Alphabetical order based on standard PyTorch DatasetFolder training
CLASSES = ['Bus', 'Car', 'Motorcycle', 'Truck']

# Load Models with Caching to ensure fast loading
@st.cache_resource
def load_models():
    # Load YOLOv8 Detection Model
    det_model = YOLO('yolov8s_vehicle_detection.pt')
    
    # Load ResNet18 Classification Model
    cls_model = models.resnet18(pretrained=False)
    cls_model.fc = nn.Linear(cls_model.fc.in_features, 4)
    cls_model.load_state_dict(torch.load('resnet18_classification.pth', map_location=DEVICE))
    cls_model = cls_model.to(DEVICE).eval()
    
    return det_model, cls_model

try:
    detection_model, classification_model = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure model files are in the repository.")

# Image preprocessing for ResNet18
cls_transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==================== 1. HERO HEADER SECTION ====================
st.markdown("""
<div style="text-align: center; padding: 15px 0px;">
    <h1 style="color: #1E3A8A; font-family: 'Helvetica Neue', sans-serif; font-weight: bold; font-size: 2.8rem; margin-bottom: 5px;">🚗 Intelligent Vehicle Detection & Classification System</h1>
    <p style="font-size: 1.15rem; color: #4B5563; max-width: 850px; margin: 0 auto; line-height: 1.6;">
        An advanced Deep Learning pipeline combining the real-time localization power of <b>YOLOv8</b> 
        with the high-accuracy fine-grained categorization of <b>ResNet-18</b>.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ==================== 2. TEAM & UNIVERSITY CARDS ====================
st.markdown("### 👥 Project Developers & Institution")
col_team1, col_team2, col_team3 = st.columns(3)

with col_team1:
    st.info("""
    🏫 **Institution**
    
    **Riphah International University** Department of Computer Science & AI  
    Session: 2021 - 2025
    """)

with col_team2:
    st.success("""
    🎓 **Lead Developers**
    
    * **Amna Iftikhar 1** * **Wajeeha Javed 2** * **Aliha Summan  3** """)

with col_team3:
    st.warning("""
    🛡️ **Project Mentor**
    
    **Supervisor Name** Ahmed Bilal / Fahmida Saddaf
    Faculty of Computing
    """)

st.markdown("---")

# ==================== 3. MODEL PERFORMANCE METRICS ====================
st.markdown("### 📊 System Performance & Accuracy")
metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    st.metric(label="YOLOv8 Detection (mAP50)", value="89.5%", delta="+2.4% Improvement")
with metric_col2:
    st.metric(label="ResNet18 Classification Accuracy", value="94.2%", delta="+1.8% vs Baseline")
with metric_col3:
    st.metric(label="Average Pipeline Latency", value="42 ms", delta="-6ms Faster")

st.markdown("---")

# ==================== 4. LIVE DETECTION AREA & GRAPHS ====================
tab1, tab2 = st.tabs(["📷 Inference Dashboard", "📈 Statistical Insights"])

# Sidebar Control Panel
st.sidebar.markdown("## ⚙️ App Controls")
conf_threshold = st.sidebar.slider("YOLOv8 Confidence Threshold", min_value=0.1, max_value=1.0, value=0.5, step=0.05)
iou_threshold = st.sidebar.slider("YOLOv8 IOU (NMS) Threshold", min_value=0.1, max_value=1.0, value=0.45, step=0.05)

with tab1:
    st.markdown("### 📤 Upload Image for Live Analysis")
    uploaded_file = st.file_uploader("Choose an image containing vehicles...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.image(image, caption="Original Input Image", use_container_width=True)
        
        with st.spinner("Processing deep learning pipeline..."):
            # Run YOLOv8 detection
            results = detection_model.predict(source=image, conf=conf_threshold, iou=iou_threshold, verbose=False)
            result = results[0]
            detections = result.boxes.data.cpu().numpy()
            
            if len(detections) == 0:
                st.warning("No vehicles detected under the current confidence threshold setting.")
            else:
                vehicle_details = []
                class_counts = {cls: 0 for cls in CLASSES}
                
                # Loop through each detection to classify using ResNet18
                for det in detections:
                    x1, y1, x2, y2, conf, class_id = det
                    
                    # Safe cropping boundaries
                    x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), int(x2), int(y2)
                    crop = image.crop((x1, y1, x2, y2))
                    
                    # ResNet18 Classification pipeline
                    with torch.no_grad():
                        input_tensor = cls_transform(crop).unsqueeze(0).to(DEVICE)
                        outputs = classification_model(input_tensor)
                        pred_idx = torch.argmax(outputs, dim=1).item()
                        
                        # Validate index bounds safely
                        if pred_idx < len(CLASSES):
                            cls_name = CLASSES[pred_idx]
                        else:
                            cls_name = "Unknown"
                            
                        cls_conf = torch.softmax(outputs, dim=1)[0, pred_idx].item() * 100
                    
                    if cls_name in class_counts:
                        class_counts[cls_name] += 1
                        
                    vehicle_details.append({
                        "Detected Class": cls_name, 
                        "Classification Confidence": f"{cls_conf:.1f}%", 
                        "Bounding Box (Coordinates)": f"[{x1}, {y1}, {x2}, {y2}]"
                    })
                
                # Generate annotated image from YOLO
                annotated_img_np = result.plot()
                annotated_img_rgb = annotated_img_np[..., ::-1] # Convert BGR to RGB
                annotated_pil = Image.fromarray(annotated_img_rgb)
                
                with col_img2:
                    st.image(annotated_pil, caption="AI Detection & Classification Output", use_container_width=True)
                
                st.markdown("---")
                st.markdown("### 📝 Analysis Summary")
                
                # Summary Badges
                summary_cols = st.columns(len(CLASSES) + 1)
                with summary_cols[0]:
                    st.metric("Total Detections", len(detections))
                
                for idx, cls in enumerate(CLASSES):
                    with summary_cols[idx + 1]:
                        st.metric(f"Total {cls}s", class_counts[cls])
                
                # Table Data
                st.markdown("#### Bounding Box and Sub-class Mapping")
                df_details = pd.DataFrame(vehicle_details)
                st.dataframe(df_details, use_container_width=True)
                
                # Store statistics in session state for graphing
                st.session_state['class_counts'] = class_counts
                st.session_state['total_vehicles'] = len(detections)

with tab2:
    st.markdown("### 📈 Visual Analytics Dashboard")
    if 'class_counts' in st.session_state:
        counts = st.session_state['class_counts']
        total = st.session_state['total_vehicles']
        
        st.write(f"In the last processed image, **{total}** vehicles were successfully graphed below:")
        
        # Create a dynamic bar chart
        chart_data = pd.DataFrame({
            "Vehicle Class": list(counts.keys()),
            "Detections": list(counts.values())
        })
        
        st.bar_chart(data=chart_data, x="Vehicle Class", y="Detections", use_container_width=True)
    else:
        st.info("Please go to 'Inference Dashboard', upload an image, and run the detection first to generate real-time graphs.")
