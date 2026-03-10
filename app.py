import streamlit as st
from roboflow import Roboflow
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Vision Industrielle - Usinage",
    page_icon="🔧",
    layout="wide"
)

# ❌ TES PARAMÈTRES (IDs initiaux conservés)
API_KEY = "9FcisW7nvl380crhBt6e"
ID_CLASSIFICATION = "usinage-1uqck"
ID_DETECTION = "test-pz2em-ghj7h"

# --- 2. FONCTION DE CHARGEMENT DU MODÈLE ---
@st.cache_resource
def load_roboflow_model(api_key, project_id, version):
    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace().project(project_id)
        model = project.version(version).model
        return model
    except Exception as e:
        st.error(f"❌ Erreur Roboflow sur {project_id} (Version {version}) : {e}")
        return None

# --- 3. BARRE LATÉRALE (SIDEBAR) ---
st.sidebar.title("🎮 Panneau de Contrôle")

# Choix de la tâche
task_type = st.sidebar.radio(
    "Action à réaliser :",
    ["Détection d'objets", "Classification d'image"]
)

# Choix de la source
app_mode = st.sidebar.selectbox(
    "Source de données", 
    ["Image Unique", "Dossier d'Images", "Vidéo"]
)

st.sidebar.divider()
st.sidebar.subheader("Réglages Modèle")
# On laisse le choix de la version au cas où la v1 n'est pas encore entraînée
model_version = st.sidebar.number_input("Version du modèle", min_value=1, value=1)
conf_threshold = st.sidebar.slider("Confiance (%)", 0, 100, 40)

# Sélection automatique de l'ID selon le choix
current_id = ID_DETECTION if task_type == "Détection d'objets" else ID_CLASSIFICATION

# Chargement du modèle
model = load_roboflow_model(API_KEY, current_id, model_version)

# --- 4. INTERFACE PRINCIPALE ---
st.title(f"🛠️ Analyse : {task_type}")
st.info(f"Projet : `{current_id}` | Version : `{model_version}`")

if model is None:
    st.warning(f"⚠️ Le modèle `{current_id}` n'est pas accessible en version {model_version}.")
    st.write("Vérifiez sur Roboflow que vous avez bien cliqué sur **'Generate'** puis **'Train'**.")
else:
    # --- MODE : IMAGE UNIQUE ---
    if app_mode == "Image Unique":
        st.header("📸 Analyse Image")
        file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        
        if file and st.button("Lancer l'analyse"):
            img = Image.open(file)
            prediction = model.predict(np.array(img), confidence=conf_threshold)
            
            col1, col2 = st.columns(2)
            col1.image(img, caption="Original", use_container_width=True)
            
            if task_type == "Détection d'objets":
                prediction.save("result.jpg")
                col2.image("result.jpg", caption="Résultat Détection", use_container_width=True)
            else:
                res_json = prediction.json()
                label = res_json['predictions'][0]['class'] if res_json['predictions'] else "Inconnu"
                st.success(f"Résultat Classification : **{label}**")
                st.json(res_json)

    # --- MODE : DOSSIER D'IMAGES ---
    elif app_mode == "Dossier d'Images":
        st.header("📁 Analyse par Lot")
        files = st.file_uploader("Upload Dossier (Images)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        
        if files and st.button("Analyser le lot"):
            cols = st.columns(3)
            for i, f in enumerate(files):
                img = Image.open(f)
                pred = model.predict(np.array(img), confidence=conf_threshold)
                
                if task_type == "Détection d'objets":
                    pred.save(f"batch_{i}.jpg")
                    cols[i % 3].image(f"batch_{i}.jpg", caption=f.name, use_container_width=True)
                else:
                    label = pred.json()['predictions'][0]['class'] if pred.json()['predictions'] else "N/A"
                    cols[i % 3].image(img, caption=f"{f.name} -> {label}", use_container_width=True)

    # --- MODE : VIDÉO ---
    elif app_mode == "Vidéo":
        st.header("🎥 Analyse Vidéo")
        v_file = st.file_uploader("Upload Vidéo", type=["mp4", "avi", "mov"])
        
        if v_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(v_file.read())
            vf = cv2.VideoCapture(tfile.name)
            st_frame = st.empty()
            
            if st.button("Démarrer la vidéo"):
                while vf.isOpened():
                    ret, frame = vf.read()
                    if not ret: break
                    
                    # Redimensionnement pour fluidité
                    frame = cv2.resize(frame, (640, 480))
                    pred = model.predict(frame, confidence=conf_threshold)
                    
                    if task_type == "Détection d'objets":
                        pred.save("v_tmp.jpg")
                        st_frame.image("v_tmp.jpg", use_container_width=True)
                    else:
                        label = pred.json()['predictions'][0]['class'] if pred.json()['predictions'] else "..."
                        cv2.putText(frame, f"CLASSE: {label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        st_frame.image(frame, channels="BGR", use_container_width=True)
                vf.release()
