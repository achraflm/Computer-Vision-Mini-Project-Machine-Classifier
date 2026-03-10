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

# ❌ TES PARAMÈTRES (Clé API et IDs Projets)
API_KEY = "9FcisW7nvl380crhBt6e"
ID_CLASSIFICATION = "usinage-1uqck"
ID_DETECTION = "test-pz2em-ghj7h"

# --- 2. FONCTION DE CHARGEMENT DU MODÈLE ---
@st.cache_resource
def load_roboflow_model(api_key, project_id, version):
    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace().project(project_id)
        # On tente de charger la version spécifiée
        model = project.version(version).model
        return model
    except Exception as e:
        st.error(f"❌ Erreur Roboflow sur {project_id} (Version {version}) : {e}")
        return None

# --- 3. BARRE LATÉRALE (NAVIGATION & RÉGLAGES) ---
st.sidebar.title("🎮 Contrôle de l'App")

# Choix de la tâche
task_type = st.sidebar.radio(
    "Quelle tâche effectuer ?",
    ["Détection d'objets", "Classification d'image"]
)

# Choix du mode d'entrée
app_mode = st.sidebar.selectbox(
    "Source de données", 
    ["Image Unique", "Dossier d'Images", "Vidéo"]
)

# Réglages du modèle
st.sidebar.divider()
st.sidebar.subheader("Configuration du Modèle")
model_version = st.sidebar.number_input("Numéro de Version (Roboflow)", min_value=1, value=1)
conf_threshold = st.sidebar.slider("Seuil de Confiance (%)", 0, 100, 40)

# Détermination de l'ID projet en fonction de la tâche
current_project_id = ID_DETECTION if task_type == "Détection d'objets" else ID_CLASSIFICATION

# Chargement dynamique du modèle
model = load_roboflow_model(API_KEY, current_project_id, model_version)

# --- 4. INTERFACE PRINCIPALE ---
st.title(f"🛠️ Dashboard : {task_type}")
st.info(f"Projet actif : `{current_project_id}` | Version : `{model_version}`")

if model is None:
    st.warning("⚠️ Le modèle n'est pas prêt. Assurez-vous d'avoir généré une version sur Roboflow.")
    st.markdown(f"[Accéder à votre projet Roboflow](https://app.roboflow.com/{current_project_id})")
else:
    # --- MODE A : IMAGE UNIQUE ---
    if app_mode == "Image Unique":
        st.header("📸 Analyse d'une Image")
        uploaded_file = st.file_uploader("Choisir une image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            if st.button("Lancer l'analyse"):
                with st.spinner('Analyse en cours...'):
                    prediction = model.predict(np.array(image), confidence=conf_threshold)
                    
                    col1, col2 = st.columns(2)
                    col1.image(image, caption="Image Originale", use_container_width=True)
                    
                    if task_type == "Détection d'objets":
                        prediction.save("result_detect.jpg")
                        col2.image("result_detect.jpg", caption="Objets Détectés", use_container_width=True)
                    else:
                        st.subheader("Résultat Classification")
                        st.write(prediction.json())

    # --- MODE B : DOSSIER D'IMAGES ---
    elif app_mode == "Dossier d'Images":
        st.header("📁 Analyse par Lot (Dossier)")
        uploaded_files = st.file_uploader("Sélectionnez plusieurs images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        
        if uploaded_files and st.button("Analyser le groupe"):
            cols = st.columns(3)
            for i, file in enumerate(uploaded_files):
                img = Image.open(file)
                pred = model.predict(np.array(img), confidence=conf_threshold)
                
                if task_type == "Détection d'objets":
                    pred.save(f"batch_{i}.jpg")
                    cols[i % 3].image(f"batch_{i}.jpg", caption=file.name, use_container_width=True)
                else:
                    top_label = pred.json()['predictions'][0]['class'] if pred.json()['predictions'] else "Inconnu"
                    cols[i % 3].image(img, caption=f"Classe: {top_label}", use_container_width=True)

    # --- MODE C : VIDÉO ---
    elif app_mode == "Vidéo":
        st.header("🎥 Analyse Vidéo")
        video_file = st.file_uploader("Charger une vidéo", type=["mp4", "avi", "mov"])
        
        if video_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            vf = cv2.VideoCapture(tfile.name)
            st_frame = st.empty()
            
            if st.button("Démarrer la lecture"):
                while vf.isOpened():
                    ret, frame = vf.read()
                    if not ret: break
                    
                    # On réduit la taille pour accélérer le traitement si besoin
                    frame_res = cv2.resize(frame, (640, 480))
                    prediction = model.predict(frame_res, confidence=conf_threshold)
                    
                    if task_type == "Détection d'objets":
                        prediction.save("frame_now.jpg")
                        st_frame.image("frame_now.jpg", use_container_width=True)
                    else:
                        # On écrit la classe sur la frame
                        res = prediction.json()['predictions']
                        label = res[0]['class'] if res else "N/A"
                        cv2.putText(frame_res, f"CLASSE: {label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        st_frame.image(frame_res, channels="BGR", use_container_width=True)
                vf.release()
