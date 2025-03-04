import streamlit as st
import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image

# Load the ONNX model
session = ort.InferenceSession("best.onnx")

# Class labels (Update as per your dataset)
CLASSES = ["Mature Cotton", "Immature Cotton"]
CONFIDENCE_THRESHOLD = 0.3

# Function to process ONNX output
def process_output(outputs):
    detections = np.squeeze(outputs[0])  # Shape: (9, 8400)
    detections = detections.T  # Transpose to (8400, 9)
    
    boxes = detections[:, :4]  # Extract (x, y, w, h)
    scores = detections[:, 4]  # Extract confidence scores
    class_ids = np.argmax(detections[:, 5:], axis=1)  # Extract class IDs

    # Apply confidence threshold
    valid_indices = scores > CONFIDENCE_THRESHOLD
    boxes = boxes[valid_indices]
    scores = scores[valid_indices]
    class_ids = class_ids[valid_indices]
    
    return boxes, scores, class_ids

# Function to draw bounding boxes
def draw_boxes(image, boxes, scores, class_ids):
    for i, box in enumerate(boxes):
        x, y, w, h = box.astype(int)
        label = f"{CLASSES[class_ids[i]]}: {scores[i]:.2f}"
        color = (0, 255, 0)
        
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

# Streamlit UI
st.title("Cotton Maturity Detection")
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    # Convert image to model format
    input_image = cv2.resize(image_np, (640, 640))
    input_image = input_image.transpose(2, 0, 1)  # HWC to CHW
    input_image = np.expand_dims(input_image, axis=0).astype(np.float32) / 255.0
    
    # Run inference
    outputs = session.run(None, {"images": input_image})
    boxes, scores, class_ids = process_output(outputs)
    
    if len(boxes) > 0:
        detected_image = draw_boxes(image_np, boxes, scores, class_ids)
        st.image(detected_image, caption="Detected Objects", use_column_width=True)
    else:
        st.warning("No objects detected. Try another image.")
