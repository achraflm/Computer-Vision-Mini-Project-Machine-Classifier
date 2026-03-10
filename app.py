import streamlit as st
from roboflow import Roboflow
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Vision Artificielle - Usinage", layout="wide")

# Tes identifiants Roboflow
API_KEY = "9FcisW7nvl380crhBt6e" 
ID_DETECTION = "test-pz2em-ghj7h"
ID_CLASSIFICATION = "usinage-1uqck" # ID utilisé précédemment

# --- 2. FONCTION DE CHARGEMENT ---
@st.cache_resource
def load_model(project_id, version=1):
    try:
        rf = Roboflow(api_key=API_KEY)
        project = rf.workspace().project(project_id)
        return project.version(version).model
    except Exception as e:
        st.error(f"Erreur Roboflow ({project_id}): {e}")
        return None

# --- 3. BARRE LATÉRALE (NAVIGATION) ---
st.sidebar.title("🚀 Panneau de Contrôle")
task_type = st.sidebar.radio("Choisir la tâche", ["Détection d'objets", "Classification"])
app_mode = st.sidebar.selectbox("Mode d'entrée", ["Image Unique", "Dossier d'Images", "Vidéo"])
conf_level = st.sidebar.slider("Confiance (%)", 0, 100, 40)

# Sélection du bon ID projet
current_id = ID_DETECTION if task_type == "Détection d'objets" else ID_CLASSIFICATION
model = load_model(current_id)

st.title(f"🛠️ Application de {task_type}")
st.write(f"ID Projet actif : `{current_id}`")

# --- 4. LOGIQUE DE TRAITEMENT ---
if model:
    # --- MODE : IMAGE UNIQUE ---
    if app_mode == "Image Unique":
        file = st.file_uploader("Charger une image", type=['jpg', 'jpeg', 'png'])
        if file:
            img = Image.open(file)
            if st.button("Analyser"):
                prediction = model.predict(np.array(img), confidence=conf_level)
                
                col1, col2 = st.columns(2)
                col1.image(img, caption="Original", use_container_width=True)
                
                if task_type == "Détection d'objets":
                    prediction.save("res.jpg")
                    col2.image("res.jpg", caption="Détection", use_container_width=True)
                else:
                    # Pour la classification, on affiche les scores
                    st.subheader("Résultats de Classification")
                    st.write(prediction.json())

    # --- MODE : DOSSIER ---
    elif app_mode == "Dossier d'Images":
        files = st.file_uploader("Charger plusieurs images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
        if files and st.button("Lancer le lot"):
            cols = st.columns(3)
            for i, f in enumerate(files):
                img = np.array(Image.open(f))
                pred = model.predict(img, confidence=conf_level)
                if task_type == "Détection d'objets":
                    pred.save(f"batch_{i}.jpg")
                    cols[i % 3].image(f"batch_{i}.jpg", caption=f.name)
                else:
                    top_class = pred.json()['predictions'][0]['class'] if pred.json()['predictions'] else "Inconnu"
                    cols[i % 3].image(img, caption=f"Classe : {top_class}")

    # --- MODE : VIDÉO ---
    elif app_mode == "Vidéo":
        v_file = st.file_uploader("Charger une vidéo", type=['mp4', 'mov', 'avi'])
        if v_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(v_file.read())
            vf = cv2.VideoCapture(tfile.name)
            st_frame = st.empty()
            
            if st.button("Démarrer l'analyse"):
                while vf.isOpened():
                    ret, frame = vf.read()
                    if not ret: break
                    
                    pred = model.predict(frame, confidence=conf_level)
                    if task_type == "Détection d'objets":
                        pred.save("v_frame.jpg")
                        st_frame.image("v_frame.jpg", use_container_width=True)
                    else:
                        # Overlay texte pour classification
                        top = pred.json()['predictions'][0]['class'] if pred.json()['predictions'] else "..."
                        cv2.putText(frame, f"Classe: {top}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                        st_frame.image(frame, channels="BGR", use_container_width=True)
                vf.release()
