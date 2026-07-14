import streamlit as st
import torch
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from PIL import Image
import torchvision.transforms as transforms

# Page configuration
st.set_page_config(page_title="Vehicle Detection & Classification", layout="wide")

# 1. Models Load karne ka function
@st.cache_resource
def load_models():
    # Yahan apni Hugging Face Repo ID daalein (e.g., "AmnaIftikhar99/vehicle-detection-project")
    repo_id = "AmnaIftikhar99/vehicle-detection-project" 
    
    # Files download karna
    yolo_path = hf_hub_download(repo_id=repo_id, filename="yolov8s_vehicle_detection.pt")
    resnet_path = hf_hub_download(repo_id=repo_id, filename="resnet18_classification.pth")
    
    # Models load karna
    model_yolo = YOLO(yolo_path)
    model_resnet = torch.load(resnet_path, map_location=torch.device('cpu'))
    model_resnet.eval()
    
    return model_yolo, model_resnet

st.title("🚗 Vehicle Detection & Classification App")
st.write("Image upload karein aur vehicles detect aur classify karein!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Model load karein
    model_yolo, model_resnet = load_models()
    
    # Image display karein
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    st.write("Processing...")
    
    # 1. Detection (YOLO)
    results = model_yolo(image)
    st.image(results[0].plot(), caption='Detected Vehicles')
    
    st.success("Analysis Complete!")
