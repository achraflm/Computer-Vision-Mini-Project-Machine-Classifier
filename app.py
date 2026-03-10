import streamlit as st
from roboflow import Roboflow
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Détection Industrielle - Usinage",
    page_icon="🛠️",
    layout="wide"
)

# ❌ TES PARAMÈTRES (Vérifie bien la version sur Roboflow)
API_KEY = "9FcisW7nvl380crhBt6e"
PROJECT_ID = "usinage-1uqck"
MODEL_VERSION = 1 

# --- 2. FONCTION DE CHARGEMENT DU MODÈLE ---
@st.cache_resource
def load_roboflow_model(api_key, project_id, version):
    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace().project(project_id)
        model = project.version(version).model
        return model
    except Exception as e:
        st.error(f"Erreur de connexion à Roboflow : {e}")
        return None

# Initialisation du modèle
model = load_roboflow_model(API_KEY, PROJECT_ID, MODEL_VERSION)

# --- 3. BARRE LATÉRALE (SIDEBAR) ---
st.sidebar.image("https://roboflow.com/images/roboflow-logo.png", width=200)
st.sidebar.title("Configuration")

app_mode = st.sidebar.selectbox(
    "Mode de détection", 
    ["Image Unique", "Dossier d'Images", "Vidéo"]
)

conf_threshold = st.sidebar.slider("Seuil de Confiance (%)", 0, 100, 40) / 100
st.sidebar.info(f"Projet : {PROJECT_ID}\nVersion : {MODEL_VERSION}")

# --- 4. LOGIQUE PRINCIPALE ---

if model is None:
    st.error("⚠️ Le modèle n'a pas pu être chargé. Vérifiez votre clé API et l'ID du projet.")
else:
    # --- MODE : IMAGE UNIQUE ---
    if app_mode == "Image Unique":
        st.header("📸 Analyse d'une Image Unique")
        uploaded_file = st.file_uploader("Choisir une image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            if st.button("Lancer la détection"):
                with st.spinner('Détection en cours...'):
                    # Prédiction
                    prediction = model.predict(img_array, confidence=conf_threshold * 100)
                    prediction.save("result.jpg")
                    
                    # Affichage
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image, caption="Image Originale", use_container_width=True)
                    with col2:
                        st.image("result.jpg", caption="Résultat de Détection", use_container_width=True)
                    
                    st.subheader("Données JSON")
                    st.json(prediction.json())

    # --- MODE : DOSSIER D'IMAGES ---
    elif app_mode == "Dossier d'Images":
        st.header("📁 Analyse par Lot (Dossier)")
        uploaded_files = st.file_uploader("Sélectionnez plusieurs images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        
        if uploaded_files and st.button("Analyser le dossier"):
            st.write(f"Analyse de {len(uploaded_files)} images...")
            cols = st.columns(3) # Affichage sur 3 colonnes
            
            for i, file in enumerate(uploaded_files):
                img = Image.open(file)
                pred = model.predict(np.array(img), confidence=conf_threshold * 100)
                pred.save(f"temp_{i}.jpg")
                cols[i % 3].image(f"temp_{i}.jpg", caption=file.name, use_container_width=True)

    # --- MODE : VIDÉO ---
    elif app_mode == "Vidéo":
        st.header("🎥 Analyse Vidéo (Frame par Frame)")
        video_file = st.file_uploader("Télécharger une vidéo", type=["mp4", "avi", "mov"])
        
        if video_file is not None:
            # Créer un fichier temporaire pour la vidéo
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            
            vf = cv2.VideoCapture(tfile.name)
            st_frame = st.empty() # Cadre pour l'affichage dynamique
            
            if st.button("Démarrer l'analyse"):
                while vf.isOpened():
                    ret, frame = vf.read()
                    if not ret:
                        break
                    
                    # Détection sur la frame actuelle
                    # Note : L'appel API sur chaque frame peut être lent
                    prediction = model.predict(frame, confidence=conf_threshold * 100)
                    prediction.save("frame_res.jpg")
                    
                    # Affichage en temps réel
                    st_frame.image("frame_res.jpg", caption="Analyse Vidéo en cours", use_container_width=True)
                
                vf.release()
                st.success("Analyse terminée.")
