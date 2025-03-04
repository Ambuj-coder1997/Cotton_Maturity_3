import os
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import gdown  # Ensure gdown is in your requirements.txt

st.title("Custom CNN Detection for Cotton Maturity Stages")

# Sidebar: Adjust detection parameters
st.sidebar.header("Detection Parameters")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

# Define the model file name and the Google Drive URL.
MODEL_FILE = "model.keras"
# Extracted file ID from your provided link.
GOOGLE_DRIVE_FILE_ID = "1buaRpIBhS3dnQtXs6ll16NBfOzjIYMAj"
GOOGLE_DRIVE_URL = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"

def download_model_if_needed():
    if not os.path.exists(MODEL_FILE):
        st.write("Model file not found locally. Downloading from Google Drive...")
        with st.spinner("Downloading model from Google Drive..."):
            gdown.download(GOOGLE_DRIVE_URL, MODEL_FILE, quiet=False)
        st.write("Download complete.")
    else:
        st.write("Model file found locally.")

@st.cache_resource
def load_model_keras():
    download_model_if_needed()
    # Diagnostic: List current directory contents to verify the model file's presence.
    st.write("Current directory files:", os.listdir('.'))
    if not os.path.exists(MODEL_FILE):
        st.error("Model file still does not exist after download attempt.")
        return None
    model = tf.keras.models.load_model(MODEL_FILE)
    return model

model = load_model_keras()

# File uploader for selecting an image for detection.
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image: resize to 256x256 and normalize.
    image_resized = image.resize((256, 256))
    image_array = np.array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    # Run inference: the model outputs a classification and bounding box prediction.
    class_pred, bbox_pred = model.predict(image_array)
    class_idx = np.argmax(class_pred[0])
    confidence = class_pred[0][class_idx]
    
    if confidence < conf_threshold:
        st.write("Detection confidence below threshold.")
    else:
        # Convert the normalized bounding box [cx, cy, w, h] to original image scale.
        width, height = image.size
        cx, cy, bw, bh = bbox_pred[0]
        cx, cy, bw, bh = cx * width, cy * height, bw * width, bh * height
        x1 = int(cx - bw / 2)
        y1 = int(cy - bh / 2)
        x2 = int(cx + bw / 2)
        y2 = int(cy + bh / 2)
        
        # Draw bounding box and label on the image.
        image_cv = np.array(image)
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            image_cv,
            f"Class {class_idx} ({confidence:.2f})",
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2,
        )
        st.image(image_cv, caption="Detection Results", use_column_width=True)
