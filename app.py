import streamlit as st
from roboflow import Roboflow
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Classification Industrielle - Achraf", layout="wide", page_icon="⚙️")

# Tes paramètres fixes
API_KEY = "9FcisW7nvl380crhBt6e"
PROJECT_ID = "usinage-1uqck"

# --- 2. CHARGEMENT DU MODÈLE ---
@st.cache_resource
def load_model(version):
    try:
        rf = Roboflow(api_key=API_KEY)
        project = rf.workspace().project(PROJECT_ID)
        return project.version(version).model
    except Exception as e:
        st.error(f"Erreur Roboflow : {e}")
        return None

# --- 3. BARRE LATÉRALE (MENU) ---
st.sidebar.title("🛠️ Menu Usinage")
mode = st.sidebar.selectbox("Choisir le mode", ["Image Unique", "Dossier d'Images", "Vidéo"])
version_n = st.sidebar.number_input("Version du modèle", min_value=1, value=2) # Par défaut V2

model = load_model(version_n)

# --- 4. INTERFACE PRINCIPALE ---
st.title("⚙️ Système de Classification de Pièces")
st.write(f"Projet actif : `{PROJECT_ID}` | Version : `{version_n}`")

if model:
    # --- MODE 1 : IMAGE UNIQUE ---
    if mode == "Image Unique":
        st.header("📸 Analyse d'une pièce")
        file = st.file_uploader("Charger une image", type=['jpg', 'jpeg', 'png'])
        
        if file:
            img = Image.open(file)
            col1, col2 = st.columns(2)
            col1.image(img, caption="Image source", use_container_width=True)
            
            if col2.button("🔍 Identifier la pièce"):
                with st.spinner('Analyse...'):
                    # Sauvegarde temporaire indispensable pour la classification
                    temp_p = "predict_temp.jpg"
                    img.save(temp_p)
                    
                    # Prediction
                    res = model.predict(temp_p).json()
                    
                    if res['predictions']:
                        top = res['predictions'][0]
                        col2.success(f"### Résultat : {top['class']}")
                        col2.metric("Confiance", f"{top['confidence']:.2%}")
                        col2.json(res)
                    else:
                        col2.warning("Aucune classe identifiée.")
# --- ANALYSE SÉCURISÉE DES RÉSULTATS ---
                res = model.predict(temp_p).json()
                
                # On vérifie si on a des prédictions
                if 'predictions' in res and len(res['predictions']) > 0:
                    # On prend la meilleure prédiction
                    top = res['predictions'][0]
                    
                    # On cherche le nom de la classe (parfois c'est 'class', parfois 'label')
                    nom_piece = top.get('class', top.get('label', 'Inconnu'))
                    # On cherche la confiance (parfois c'est 'confidence', parfois 'score')
                    score = top.get('confidence', top.get('score', 0))
                    
                    col2.success(f"### Résultat : {nom_piece}")
                    col2.metric("Confiance", f"{score:.2%}")
                
                # Optionnel : Affiche le JSON brut en dessous pour débugger si besoin
                with st.expander("Détails techniques (JSON)"):
                    st.json(res)
    # --- MODE 2 : DOSSIER D'IMAGES ---
    elif mode == "Dossier d'Images":
        st.header("📁 Analyse par lot")
        files = st.file_uploader("Sélectionner plusieurs images", type=['jpg', 'png'], accept_multiple_files=True)
        
        if files and st.button("Lancer l'analyse du lot"):
            st.write(f"Traitement de {len(files)} images...")
            grid = st.columns(3)
            for i, f in enumerate(files):
                img = Image.open(f)
                temp_p = f"batch_{i}.jpg"
                img.save(temp_p)
                
                prediction = model.predict(temp_p).json()
                label = prediction['predictions'][0]['class'] if prediction['predictions'] else "Inconnu"
                
                grid[i % 3].image(img, caption=f"{f.name} ➡️ {label}", use_container_width=True)

    # --- MODE 3 : VIDÉO ---
    elif mode == "Vidéo":
        st.header("🎥 Analyse vidéo en temps réel")
        v_file = st.file_uploader("Charger une vidéo", type=['mp4', 'avi', 'mov'])
        
        if v_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(v_file.read())
            vf = cv2.VideoCapture(tfile.name)
            st_frame = st.empty()
            
            if st.button("Démarrer la lecture"):
                while vf.isOpened():
                    ret, frame = vf.read()
                    if not ret: break
                    
                    # Traitement de la frame
                    cv2.imwrite("frame_temp.jpg", frame)
                    prediction = model.predict("frame_temp.jpg").json()
                    
                    if prediction['predictions']:
                        label = prediction['predictions'][0]['class']
                        conf = prediction['predictions'][0]['confidence']
                        # On écrit sur l'image
                        text = f"{label} ({conf:.1%})"
                        cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    st_frame.image(frame, channels="BGR", use_container_width=True)
                vf.release()

else:
    st.error("❌ Impossible de charger le modèle. Vérifie que la version est 'Trained' sur Roboflow.")
