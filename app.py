#!/usr/bin/env python3
import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Define classes and device
CLASSES = ['Car', 'Bus', 'Truck', 'Motorcycle']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load models
# Ensure these files are in the same folder as app.py
detection_model = YOLO('yolov8s_vehicle_detection.pt')
classification_model = models.resnet18(pretrained=False)
classification_model.fc = nn.Linear(classification_model.fc.in_features, 4)
classification_model.load_state_dict(torch.load('resnet18_classification.pth', map_location=DEVICE))
classification_model = classification_model.to(DEVICE).eval()

# Transformation for classification
cls_transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def process_image(image):
    if image is None: return None, "No image provided", ""
    
    # Run detection
    results = detection_model.predict(source=image, conf=0.5, iou=0.45, verbose=False)
    result = results[0]
    detections = result.boxes.data.cpu().numpy()
    
    if len(detections) == 0: return image, "No vehicles detected", ""
    
    # Process crops and classify
    vehicle_details = []
    class_counts = {cls: 0 for cls in CLASSES}
    
    for det in detections:
        x1, y1, x2, y2, conf, class_id = det
        crop = image.crop((int(x1), int(y1), int(x2), int(y2)))
        
        with torch.no_grad():
            outputs = classification_model(cls_transform(crop).unsqueeze(0).to(DEVICE))
            confidence = torch.softmax(outputs, dim=1)
            pred = torch.argmax(confidence, dim=1).item()
            cls_name = CLASSES[pred]
            
        class_counts[cls_name] += 1
        vehicle_details.append(f"{cls_name} ({confidence[0, pred].item()*100:.1f}%)")
        
    annotated_pil = Image.fromarray(result.plot()[..., ::-1])
    stats_text = f"Total Vehicles: {len(detections)}
" + "
".join([f"{k}: {v}" for k, v in class_counts.items() if v > 0])
    
    return annotated_pil, stats_text, "
".join(vehicle_details)

# Create Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# Vehicle Detection & Classification System")
    with gr.Row():
        image_input = gr.Image(type="pil")
        output_image = gr.Image(type="pil")
    submit_btn = gr.Button("Run Pipeline")
    stats_output = gr.Textbox(label="Statistics")
    details_output = gr.Textbox(label="Vehicle Details")
    submit_btn.click(fn=process_image, inputs=[image_input], outputs=[output_image, stats_output, details_output])

if __name__ == "__main__":
    app.launch()
