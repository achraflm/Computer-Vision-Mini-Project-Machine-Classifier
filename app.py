import streamlit as st
from roboflow import Roboflow
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="IA Usinage - Détection V1", layout="wide", page_icon="🔧")

API_KEY = "9FcisW7nvl380crhBt6e"
PROJECT_ID = "usinage-detection" 
VERSION = 1  # Fixé sur la version 1 comme demandé

@st.cache_resource
def load_model():
    try:
        rf = Roboflow(api_key=API_KEY)
        project = rf.workspace().project(PROJECT_ID)
        return project.version(VERSION).model
    except Exception as e:
        st.error(f"Erreur : Impossible de charger la V1 du projet {PROJECT_ID}. {e}")
        return None

# --- 2. BARRE LATÉRALE ---
st.sidebar.title("🎮 Dashboard de Contrôle")
task_type = st.sidebar.radio("Mode d'analyse", ["Détection d'objets (Boîtes)", "Classification (Texte)"])
input_mode = st.sidebar.selectbox("Source de données", ["Image Unique", "Dossier d'Images", "Vidéo"])

st.sidebar.divider()
threshold = st.sidebar.slider("Seuil de Confiance (%)", 0, 100, 50)

model = load_model()

# --- 3. INTERFACE PRINCIPALE ---
st.title(f"🛠️ Système Expert Vision : {task_type}")
st.write(f"ID Projet : `{PROJECT_ID}` | Version : `{VERSION}`")

[Image of object detection vs image classification]

if model:
    # --- MODE 1 : IMAGE UNIQUE ---
    if input_mode == "Image Unique":
        file = st.file_uploader("Charger une image", type=['jpg', 'jpeg', 'png'])
        if file:
            img = Image.open(file)
            col1, col2 = st.columns(2)
            col1.image(img, caption="Image originale", use_container_width=True)
            
            if col2.button("🔍 Lancer l'Analyse"):
                temp_p = "detect_temp.jpg"
                img.save(temp_p)
                
                # Exécution du modèle (On multiplie le seuil par 1 pour l'API)
                prediction = model.predict(temp_p, confidence=threshold)
                
                if "Détection" in task_type:
                    prediction.save("res_plot.jpg")
                    col2.image("res_plot.jpg", caption="Résultat avec localisation")
                else:
                    preds = prediction.json().get('predictions', [])
                    if preds:
                        top = preds[0]
                        col2.success(f"### Pièce identifiée : **{top['class']}**")
                        col2.metric("Indice de confiance", f"{top['confidence']:.2%}")
                    else:
                        col2.warning("Aucune détection concluante pour ce seuil.")
                
                with st.expander("📄 Voir les données JSON"):
                    st.json(prediction.json())

    # --- MODE 2 : DOSSIER D'IMAGES ---
    elif input_mode == "Dossier d'Images":
        st.header("📁 Analyse par lot (Dossier)")
        files = st.file_uploader("Upload Dossier", accept_multiple_files=True)
        if files and st.button("Lancer l'analyse du lot"):
            grid = st.columns(3)
            for i, f in enumerate(files):
                img = Image.open(f)
                path = f"batch_{i}.jpg"
                img.save(path)
                
                prediction = model.predict(path, confidence=threshold)
                
                if "Détection" in task_type:
                    prediction.save(f"out_{i}.jpg")
                    grid[i % 3].image(f"out_{i}.jpg", caption=f.name)
                else:
                    preds = prediction.json().get('predictions', [])
                    label = preds[0]['class'] if preds else "Inconnu"
                    grid[i % 3].image(img, caption=f"{f.name} : {label}")

    # --- MODE 3 : VIDÉO ---
    elif input_mode == "Vidéo":
        st.header("🎥 Analyse Vidéo en temps réel")
        v_file = st.file_uploader("Charger une vidéo", type=['mp4', 'avi'])
        if v_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(v_file.read())
            vf = cv2.VideoCapture(tfile.name)
            placeholder = st.empty()
            
            last_text = "Calcul..."
            f_count = 0

            if st.button("Démarrer la vidéo"):
                while vf.isOpened():
                    ret, frame = vf.read()
                    if not ret: break
                    f_count += 1
                    
                    # Optimisation pour fluidité (1 image sur 5)
                    if f_count % 5 == 0:
                        cv2.imwrite("frame.jpg", frame)
                        prediction = model.predict("frame.jpg", confidence=threshold)
                        
                        if "Détection" in task_type:
                            prediction.save("frame_res.jpg")
                            frame = cv2.imread("frame_res.jpg")
                        else:
                            preds = prediction.json().get('predictions', [])
                            last_text = f"{preds[0]['class']} ({preds[0]['confidence']:.1%})" if preds else "..."
                    
                    if "Classification" in task_type:
                        cv2.putText(frame, last_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    
                    placeholder.image(frame, channels="BGR", use_container_width=True)
                vf.release()
