import streamlit as st
from roboflow import Roboflow
from PIL import Image
import numpy as np

# --- 1. SETUP ---
API_KEY = "9FcisW7nvl380crhBt6e"
PROJECT_ID = "usinage-1uqck" 
VERSION = 1 

st.set_page_config(page_title="Usinage Classifier", page_icon="🔧")
st.title("🔧 Usinage Part Classifier")

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        rf = Roboflow(api_key=API_KEY)
        project = rf.workspace().project(PROJECT_ID)
        # This line fails if version isn't "Trained"
        return project.version(VERSION).model
    except Exception as e:
        st.error(f"❌ Connection Error: {e}")
        return None

model = load_model()

# --- 3. UI ---
uploaded_file = st.file_uploader("Upload a part image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Target Image", use_container_width=True)
    
    if st.button("🚀 Identify Part"):
        if model:
            with st.spinner('Analyzing...'):
                # 💡 CLASSIFICATION SYNTAX
                # We use .predict() but we don't use .save() because there are no boxes!
                prediction = model.predict(np.array(image)).json()
                
                if "predictions" in prediction and len(prediction["predictions"]) > 0:
                    top_prediction = prediction["predictions"][0]
                    label = top_prediction["class"]
                    confidence = top_prediction["confidence"]
                    
                    st.metric(label="Detected Part", value=label)
                    st.progress(confidence)
                    st.write(f"Confidence Score: {confidence:.2%}")
                else:
                    st.warning("No classification result found.")
        else:
            st.error("Model not loaded. Check your Roboflow dashboard.")
