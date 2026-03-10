import streamlit as st
from roboflow import Roboflow
import cv2
import numpy as np
from PIL import Image
import os

# --- 1. CONFIGURATION ---
# IMPORTANT: Put your API Key in Streamlit Secrets or an Environment Variable
ROBOFLOW_API_KEY = st.sidebar.text_input("Enter API KEY", type="password")
PROJECT_ID = "usinage-1uqck" # Your project ID
MODEL_VERSION = 1 # Change this if you have multiple versions

st.title("🛠️ Usinage Detection Dashboard")
st.write(f"Project ID: {PROJECT_ID}")

# --- 2. ROBOFLOW LOGIC ---
if ROBOFLOW_API_KEY:
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace().project(PROJECT_ID)
    model = project.version(MODEL_VERSION).model
else:
    st.warning("Please enter your Roboflow API Key in the sidebar to begin.")

# --- 3. FRONT-END UI ---
uploaded_file = st.file_uploader("Upload a machining part image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and ROBOFLOW_API_KEY:
    # Convert uploaded file to an image OpenCV can read
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("🔍 Run Detection"):
        with st.spinner('Analyzing...'):
            # Run Inference
            # We save temporarily because Roboflow API likes file paths or numpy arrays
            results = model.predict(img_array, confidence=40, overlap=30).json()
            
            # --- 4. DISPLAY RESULTS ---
            predictions = results['predictions']
            
            if not predictions:
                st.info("No objects detected.")
            else:
                # Roboflow can also return a plotted image
                # But here is how you see the raw data:
                st.write(f"Detected {len(predictions)} objects:")
                st.json(predictions)
                
                # Optional: Use Roboflow's built-in save to show the image with boxes
                model.predict(img_array).save("prediction.jpg")
                st.image("prediction.jpg", caption="Detection Results", use_container_width=True)
