import streamlit as st
from roboflow import Roboflow
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="IA Industrielle Achraf", layout="wide", page_icon="🛠️")

API_KEY = "9FcisW7nvl380crhBt6e"
ID_CLASSIFICATION = "usinage-1uqck"
ID_DETECTION = "test-pz2em-ghj7h"
VERSION = 2  # Fixé sur la version 2

@st.cache_resource
def load_model(project_id):
    try:
        rf = Roboflow(api_key=API_KEY)
        project = rf.workspace().project(project_id)
        return project.version(VERSION).model
    except Exception as e:
        st.error(f"Erreur de chargement ({project_id}) : {e}")
        return None

# --- 2. BARRE LATÉRALE ---
st.sidebar.title("🎮 Panneau de Contrôle")
task = st.sidebar.radio("Choisir la tâche", ["Classification", "Détection d'objets"])
mode = st.sidebar.selectbox("Mode d'entrée", ["Image Unique", "Dossier d'Images", "Vidéo"])

st.sidebar.divider()
st.sidebar.subheader("Réglages")
threshold = st.sidebar.slider("Seuil de Confiance (%)", 0, 100, 50) / 100

# Chargement du modèle correspondant
current_id = ID_CLASSIFICATION if task == "Classification" else ID_DETECTION
model = load_model(current_id)

# --- 3. INTERFACE PRINCIPALE ---
st.title(f"🚀 Système de {task}")
st.info(f"Projet : `{current_id}` | Version : `{VERSION}`")



if model:
    # --- MODE 1 : IMAGE UNIQUE ---
    if mode == "Image Unique":
        file = st.file_uploader("Charger une image", type=['jpg', 'jpeg', 'png'])
        if file:
            img = Image.open(file)
            col1, col2 = st.columns(2)
            col1.image(img, caption="Source", use_container_width=True)
            
            if col2.button("🔍 Analyser"):
                temp_p = "predict_temp.jpg"
                img.save(temp_p)
                
                if task == "Classification":
                    # Logique Classification (utilise 'top')
                    res = model.predict(temp_p).json()
                    preds = res.get('predictions', [])
                    if preds:
                        top = preds[0]
                        if top.get('confidence', 0) >= threshold:
                            col2.success(f"### Pièce : **{top.get('top')}**")
                            col2.metric("Confiance", f"{top.get('confidence'):.2%}")
                        else:
                            col2.warning("Résultat sous le seuil de confiance.")
                    st.json(res)
                else:
                    # Logique Détection (utilise 'class' et dessine des boîtes)
                    prediction = model.predict(temp_p, confidence=threshold*100)
                    prediction.save("res_detect.jpg")
                    col2.image("res_detect.jpg", caption="Objets Détectés")
                    st.json(prediction.json())

    # --- MODE 2 : DOSSIER D'IMAGES ---
    elif mode == "Dossier d'Images":
        files = st.file_uploader("Upload Dossier", accept_multiple_files=True)
        if files and st.button("Lancer le lot"):
            grid = st.columns(3)
            for i, f in enumerate(files):
                img = Image.open(f)
                temp_p = f"batch_{i}.jpg"
                img.save(temp_p)
                
                if task == "Classification":
                    data = model.predict(temp_p).json()
                    preds = data.get('predictions', [])
                    label = preds[0].get('top') if preds and preds[0].get('confidence', 0) >= threshold else "❌"
                    grid[i % 3].image(img, caption=f"{f.name} : {label}")
                else:
                    pred = model.predict(temp_p, confidence=threshold*100)
                    pred.save(f"out_{i}.jpg")
                    grid[i % 3].image(f"out_{i}.jpg", caption=f.name)

    # --- MODE 3 : VIDÉO ---
    elif mode == "Vidéo":
        v_file = st.file_uploader("Charger Vidéo", type=['mp4', 'avi'])
        if v_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(v_file.read())
            vf = cv2.VideoCapture(tfile.name)
            placeholder = st.empty()
            
            last_text = "Analyse..."
            f_count = 0

            if st.button("Démarrer"):
                while vf.isOpened():
                    ret, frame = vf.read()
                    if not ret: break
                    f_count += 1
                    
                    if f_count % 5 == 0:
                        cv2.imwrite("frame.jpg", frame)
                        if task == "Classification":
                            res = model.predict("frame.jpg").json()
                            preds = res.get('predictions', [])
                            if preds and preds[0].get('confidence', 0) >= threshold:
                                last_text = f"{preds[0].get('top')} ({preds[0].get('confidence'):.1%})"
                        else:
                            # Détection sur vidéo (on récupère les données JSON pour l'affichage texte)
                            res = model.predict("frame.jpg", confidence=threshold*100).json()
                            preds = res.get('predictions', [])
                            last_text = f"{len(preds)} objets détectés"
                            # Pour la détection, on peut aussi sauvegarder l'image avec boîtes
                            model.predict("frame.jpg", confidence=threshold*100).save("frame_res.jpg")
                            frame = cv2.imread("frame_res.jpg")

                    if task == "Classification":
                        cv2.putText(frame, last_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    
                    placeholder.image(frame, channels="BGR", use_container_width=True)
                vf.release()
