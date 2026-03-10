import streamlit as st
from roboflow import Roboflow
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Vision Industrielle Achraf", layout="wide", page_icon="🔧")

# Tes identifiants
API_KEY = "9FcisW7nvl380crhBt6e"
ID_CLASSIFICATION = "usinage-1uqck"
ID_DETECTION = "test-pz2em-ghj7h"

# --- 2. FONCTION DE CHARGEMENT ---
@st.cache_resource
def load_model(project_id, version):
    try:
        rf = Roboflow(api_key=API_KEY)
        project = rf.workspace().project(project_id)
        return project.version(version).model
    except Exception as e:
        st.error(f"Erreur de chargement : {e}")
        return None

# --- 3. SIDEBAR (NAVIGATION) ---
st.sidebar.title("🚀 Contrôle")
task = st.sidebar.radio("Tâche", ["Classification", "Détection d'objets"])
mode = st.sidebar.selectbox("Mode d'entrée", ["Image Unique", "Dossier d'Images", "Vidéo"])

st.sidebar.divider()
version = st.sidebar.number_input("Version du modèle", min_value=1, value=2) # V2 par défaut
conf = st.sidebar.slider("Confiance (%)", 0, 100, 40)

# Sélection du projet
target_id = ID_CLASSIFICATION if task == "Classification" else ID_DETECTION
model = load_model(target_id, version)

# --- 4. INTERFACE PRINCIPALE ---
st.title(f"🛠️ {task} - Projet `{target_id}`")

if model:
    # --- MODE IMAGE UNIQUE ---
    if mode == "Image Unique":
        file = st.file_uploader("Charger une image", type=['jpg', 'jpeg', 'png'])
        if file:
            img = Image.open(file)
            st.image(img, caption="Image source", width=400)
            
            if st.button("Lancer l'Analyse"):
                # FIX pour TypeError : Sauvegarde temporaire
                temp_path = "temp_img.jpg"
                img.save(temp_path)
                
                prediction = model.predict(temp_path, confidence=conf).json()
                
                if task == "Détection d'objets":
                    model.predict(temp_path, confidence=conf).save("res.jpg")
                    st.image("res.jpg", caption="Résultat Détection")
                else:
                    if prediction['predictions']:
                        res = prediction['predictions'][0]
                        st.success(f"Résultat : **{res['class']}** ({res['confidence']:.2%})")
                    st.json(prediction)

    # --- MODE DOSSIER ---
    elif mode == "Dossier d'Images":
        files = st.file_uploader("Charger plusieurs images", type=['jpg', 'png'], accept_multiple_files=True)
        if files and st.button("Analyser le lot"):
            cols = st.columns(3)
            for i, f in enumerate(files):
                img = Image.open(f)
                path = f"temp_{i}.jpg"
                img.save(path)
                
                pred = model.predict(path, confidence=conf).json()
                if task == "Détection d'objets":
                    model.predict(path, confidence=conf).save(f"out_{i}.jpg")
                    cols[i % 3].image(f"out_{i}.jpg", caption=f.name)
                else:
                    label = pred['predictions'][0]['class'] if pred['predictions'] else "Inconnu"
                    cols[i % 3].image(img, caption=f"{f.name}: {label}")

    # --- MODE VIDÉO ---
    elif mode == "Vidéo":
        v_file = st.file_uploader("Charger une vidéo", type=['mp4', 'mov', 'avi'])
        if v_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(v_file.read())
            vf = cv2.VideoCapture(tfile.name)
            st_frame = st.empty()
            
            if st.button("Démarrer la vidéo"):
                while vf.isOpened():
                    ret, frame = vf.read()
                    if not ret: break
                    
                    # On enregistre la frame pour éviter le TypeError
                    cv2.imwrite("frame.jpg", frame)
                    pred = model.predict("frame.jpg", confidence=conf).json()
                    
                    if task == "Détection d'objets":
                        model.predict("frame.jpg", confidence=conf).save("frame_res.jpg")
                        st_frame.image("frame_res.jpg")
                    else:
                        label = pred['predictions'][0]['class'] if pred['predictions'] else "..."
                        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        st_frame.image(frame, channels="BGR")
                vf.release()

else:
    st.error("Le modèle n'a pas pu être chargé. Vérifie ta clé API et la version sur Roboflow.")
