import streamlit as st
from roboflow import Roboflow
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="IA Usinage - Dashboard", layout="wide", page_icon="⚙️")

API_KEY = "9FcisW7nvl380crhBt6e"
PROJECT_ID = "usinage-1uqck"

@st.cache_resource
def load_model(v):
    try:
        rf = Roboflow(api_key=API_KEY)
        project = rf.workspace().project(PROJECT_ID)
        return project.version(v).model
    except Exception as e:
        st.error(f"Erreur de connexion : {e}")
        return None

# --- 2. BARRE LATÉRALE (Paramètres) ---
st.sidebar.title("🛠️ Configuration")
mode = st.sidebar.selectbox("Mode", ["Image Unique", "Dossier d'Images", "Vidéo"])
version_n = st.sidebar.number_input("Version", min_value=1, value=2)

# 🎚️ AJOUT DU SLIDER DE SEUIL (THRESHOLD)
st.sidebar.divider()
st.sidebar.subheader("Sensibilité")
threshold = st.sidebar.slider("Seuil de Confiance (%)", 0, 100, 50) / 100

model = load_model(version_n)

# --- 3. INTERFACE PRINCIPALE ---
st.title("⚙️ Classification de Pièces d'Usinage")

if model:
    # --- MODE IMAGE UNIQUE ---
    if mode == "Image Unique":
        file = st.file_uploader("Charger une image", type=['jpg', 'jpeg', 'png'])
        if file:
            img = Image.open(file)
            col1, col2 = st.columns(2)
            col1.image(img, caption="Pièce à analyser", use_container_width=True)
            
            if col2.button("🔍 Lancer l'Analyse"):
                temp_p = "temp_predict.jpg"
                img.save(temp_p)
                
                # Récupération du JSON
                response = model.predict(temp_p).json()
                preds = response.get('predictions', [])
                
                if preds:
                    top = preds[0]
                    confiance = top.get('confidence', 0)
                    classe = top.get('class', 'Inconnu')
                    
                    # ⚖️ MANIPULATION DU SEUIL
                    if confiance >= threshold:
                        col2.success(f"### Pièce identifiée : **{classe}**")
                        col2.metric("Niveau de certitude", f"{confiance:.2%}")
                    else:
                        col2.warning(f"⚠️ Résultat incertain ({confiance:.2%}). Le seuil est fixé à {threshold:.0%}.")
                    
                    # 📝 AFFICHAGE DU JSON
                    st.divider()
                    st.subheader("📄 Données brutes (JSON)")
                    st.json(response)
                else:
                    col2.error("Aucune donnée reçue du modèle.")

    # --- MODE DOSSIER ---
    elif mode == "Dossier d'Images":
        files = st.file_uploader("Upload Dossier", accept_multiple_files=True)
        if files and st.button("Analyser le lot"):
            grid = st.columns(3)
            for i, f in enumerate(files):
                img = Image.open(f)
                temp_p = f"batch_{i}.jpg"
                img.save(temp_p)
                
                res = model.predict(temp_p).json()
                preds = res.get('predictions', [])
                
                if preds:
                    top = preds[0]
                    # On n'affiche l'étiquette que si elle dépasse le seuil
                    label = top['class'] if top['confidence'] >= threshold else "❌ Sous le seuil"
                    grid[i % 3].image(img, caption=f"{f.name} : {label}")

    # --- MODE VIDÉO ---
    elif mode == "Vidéo":
        v_file = st.file_uploader("Charger Vidéo", type=['mp4', 'avi'])
        if v_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(v_file.read())
            vf = cv2.VideoCapture(tfile.name)
            st_frame = st.empty()
            
            if st.button("Démarrer"):
                while vf.isOpened():
                    ret, frame = vf.read()
                    if not ret: break
                    
                    cv2.imwrite("frame.jpg", frame)
                    res = model.predict("frame.jpg").json()
                    preds = res.get('predictions', [])
                    
                    if preds and preds[0]['confidence'] >= threshold:
                        text = f"{preds[0]['class']} ({preds[0]['confidence']:.1%})"
                        cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    st_frame.image(frame, channels="BGR", use_container_width=True)
                vf.release()
