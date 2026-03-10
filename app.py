import streamlit as st
from roboflow import Roboflow
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# --- CONFIGURATION ---
st.set_page_config(page_title="Détection Industrielle - Usinage", layout="wide")

# ❌ Tes informations (Remplies avec ce que tu m'as donné)
PRIVATE_API_KEY = "9FcisW7nvl380crhBt6e"
PROJECT_ID = "usinage-1uqck"
MODEL_VERSION = 1 

# Initialisation du modèle
@st.cache_resource
def load_model():
    rf = Roboflow(api_key=PRIVATE_API_KEY)
    project = rf.workspace().project(PROJECT_ID)
    return project.version(MODEL_VERSION).model

model = load_model()

# --- SIDEBAR (MENU) ---
st.sidebar.title("⚙️ Options de Détection")
app_mode = st.sidebar.selectbox("Choisissez le mode", ["Image Unique", "Dossier d'Images", "Vidéo"])
conf_threshold = st.sidebar.slider("Seuil de Confiance (%)", 0, 100, 40)

# --- LOGIQUE DE DÉTECTION ---

# --- MODE 1 : IMAGE UNIQUE ---
if app_mode == "Image Unique":
    st.header("📸 Analyse d'une Image")
    file = st.file_uploader("Télécharger une image", type=['jpg', 'jpeg', 'png'])
    
    if file:
        img = Image.open(file)
        img_array = np.array(img)
        
        if st.button("Lancer la détection"):
            with st.spinner("Analyse en cours..."):
                prediction = model.predict(img_array, confidence=conf_threshold)
                prediction.save("result.jpg")
                
                col1, col2 = st.columns(2)
                col1.image(img, caption="Original", use_container_width=True)
                col2.image("result.jpg", caption="Détecté", use_container_width=True)
                st.json(prediction.json())

# --- MODE 2 : DOSSIER D'IMAGES ---
elif app_mode == "Dossier d'Images":
    st.header("📁 Analyse par Lot (Batch)")
    files = st.file_uploader("Sélectionnez plusieurs images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
    
    if files and st.button("Analyser tout le dossier"):
        cols = st.columns(3) # Affichage sur 3 colonnes
        for i, file in enumerate(files):
            img = Image.open(file)
            prediction = model.predict(np.array(img), confidence=conf_threshold)
            prediction.save(f"temp_{i}.jpg")
            cols[i % 3].image(f"temp_{i}.jpg", caption=file.name, use_container_width=True)

# --- MODE 3 : VIDÉO ---
elif app_mode == "Vidéo":
    st.header("🎥 Analyse Vidéo")
    video_file = st.file_uploader("Télécharger une vidéo", type=['mp4', 'avi', 'mov'])
    
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        
        vf = cv2.VideoCapture(tfile.name)
        st_frame = st.empty() # Placeholder pour la vidéo en direct
        
        if st.button("Démarrer l'analyse vidéo"):
            while vf.isOpened():
                ret, frame = vf.read()
                if not ret: break
                
                # Inference sur la frame
                # Note: La vidéo peut être lente selon ta connexion internet
                prediction = model.predict(frame, confidence=conf_threshold)
                
                # On dessine les résultats manuellement ou on utilise .save()
                # Pour la vidéo, on affiche le JSON ou une image temporaire
                prediction.save("frame_res.jpg")
                st_frame.image("frame_res.jpg", channels="BGR")
            vf.release()
