import streamlit as st
from roboflow import Roboflow
import numpy as np
from PIL import Image

# --- CONFIGURATION ---
API_KEY = "9FcisW7nvl380crhBt6e"
# ❌ VÉRIFIE CES DEUX VALEURS DANS LE "VIEW CODE" DE ROBOFLOW
PROJECT_ID = "usinage-1uqck" 
VERSION = 1 

st.title("🛠️ Classification d'Usinage")

@st.cache_resource
def get_model():
    try:
        rf = Roboflow(api_key=API_KEY)
        project = rf.workspace().project(PROJECT_ID)
        return project.version(VERSION).model
    except Exception as e:
        st.error(f"Erreur : {e}")
        return None

model = get_model()

# --- INTERFACE ---
uploaded_file = st.file_uploader("Charger une pièce...", type=["jpg", "png", "jpeg"])

if uploaded_file and model:
    image = Image.open(uploaded_file)
    st.image(image, caption="Pièce à analyser", use_container_width=True)
    
    if st.button("🔍 Identifier la pièce"):
        # Prediction pour la classification
        prediction = model.predict(np.array(image)).json()
        
        if "predictions" in prediction and len(prediction["predictions"]) > 0:
            # On récupère le résultat principal
            res = prediction["predictions"][0]
            classe = res["class"]
            confiance = res["confidence"]
            
            st.success(f"Résultat : **{classe}** (Confiance : {confiance:.1%})")
        else:
            st.warning("Impossible d'identifier la pièce.")
