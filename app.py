import streamlit as st
from roboflow import Roboflow
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Détection Industrielle Achraf", layout="wide", page_icon="🔍")

# Paramètres uniques
API_KEY = "9FcisW7nvl380crhBt6e"
PROJECT_ID = "usinage-detection" 
VERSION = 2  # Fixé en version 2

@st.cache_resource
def load_model():
    try:
        rf = Roboflow(api_key=API_KEY)
        project = rf.workspace().project(PROJECT_ID)
        return project.version(VERSION).model
    except Exception as e:
        st.error(f"Erreur de connexion : {e}")
        return None

# --- 2. BARRE LATÉRALE ---
st.sidebar.title("🎮 Contrôle Détection")
# On propose deux vues : avec les boîtes ou juste le texte
view_type = st.sidebar.radio("Affichage", ["Boîtes de détection", "Texte uniquement"])
mode = st.sidebar.selectbox("Source d'entrée", ["Image Unique", "Dossier d'Images", "Vidéo"])

st.sidebar.divider()
threshold = st.sidebar.slider("Seuil de Confiance (%)", 0, 100, 50)

model = load_model()

# --- 3. INTERFACE PRINCIPALE ---
st.title(f"🛠️ Analyse par Détection : `{PROJECT_ID}`")

if model:
    # --- MODE IMAGE UNIQUE ---
    if mode == "Image Unique":
        file = st.file_uploader("Charger une image", type=['jpg', 'jpeg', 'png'])
        if file:
            img = Image.open(file)
            col1, col2 = st.columns(2)
            col1.image(img, caption="Image originale", use_container_width=True)
            
            if col2.button("🚀 Lancer la détection"):
                temp_p = "detect_temp.jpg"
                img.save(temp_p)
                
                # Exécution du modèle
                prediction = model.predict(temp_p, confidence=threshold)
                
                if view_type == "Boîtes de détection":
                    prediction.save("res_plot.jpg")
                    col2.image("res_plot.jpg", caption="Objets localisés")
                else:
                    preds = prediction.json().get('predictions', [])
                    if preds:
                        top = preds[0]
                        col2.success(f"### Objet détecté : **{top['class']}**")
                        col2.metric("Confiance", f"{top['confidence']:.2%}")
                    else:
                        col2.warning("Aucun objet détecté au-dessus du seuil.")
                
                with st.expander("🔍 Voir les données brutes (JSON)"):
                    st.json(prediction.json())

    # --- MODE DOSSIER ---
    elif mode == "Dossier d'Images":
        st.header("📁 Analyse par lot")
        files = st.file_uploader("Upload Dossier", accept_multiple_files=True)
        if files and st.button("Analyser tout le dossier"):
            grid = st.columns(3)
            for i, f in enumerate(files):
                img = Image.open(f)
                path = f"batch_{i}.jpg"
                img.save(path)
                
                prediction = model.predict(path, confidence=threshold)
                
                if view_type == "Boîtes de détection":
                    prediction.save(f"out_{i}.jpg")
                    grid[i % 3].image(f"out_{i}.jpg", caption=f.name)
                else:
                    preds = prediction.json().get('predictions', [])
                    label = preds[0]['class'] if preds else "❌"
                    grid[i % 3].image(img, caption=f"{f.name} : {label}")

    # --- MODE VIDÉO ---
    elif mode == "Vidéo":
        st.header("🎥 Analyse Vidéo Stabilisée")
        v_file = st.file_uploader("Charger une vidéo", type=['mp4', 'avi'])
        if v_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(v_file.read())
            vf = cv2.VideoCapture(tfile.name)
            placeholder = st.empty()
            
            f_count = 0
            last_text = "Calcul en cours..."

            if st.button("Démarrer la vidéo"):
                while vf.isOpened():
                    ret, frame = vf.read()
                    if not ret: break
                    f_count += 1
                    
                    # Analyse 1 image sur 5 pour la fluidité
                    if f_count % 5 == 0:
                        cv2.imwrite("frame.jpg", frame)
                        prediction = model.predict("frame.jpg", confidence=threshold)
                        
                        if view_type == "Boîtes de détection":
                            prediction.save("frame_res.jpg")
                            frame = cv2.imread("frame_res.jpg")
                        else:
                            preds = prediction.json().get('predictions', [])
                            last_text = f"{preds[0]['class']} ({preds[0]['confidence']:.1%})" if preds else "..."
                    
                    if view_type == "Texte uniquement":
                        cv2.putText(frame, last_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    
                    placeholder.image(frame, channels="BGR", use_container_width=True)
                vf.release()
else:
    st.error("❌ Le modèle n'a pas pu être chargé. Vérifie l'ID et la version sur Roboflow.")
