import streamlit as st
from roboflow import Roboflow
import cv2
import numpy as np
from PIL import Image
import tempfile

# --- CONFIGURATION ---
st.set_page_config(page_title="IA Usinage Stable", layout="wide")
API_KEY = "9FcisW7nvl380crhBt6e"
PROJECT_ID = "usinage-1uqck"

@st.cache_resource
def load_model(v):
    rf = Roboflow(api_key=API_KEY)
    return rf.workspace().project(PROJECT_ID).version(v).model

# --- SIDEBAR ---
st.sidebar.title("⚙️ Réglages")
version_n = st.sidebar.number_input("Version", min_value=1, value=2)
threshold = st.sidebar.slider("Seuil de Confiance (%)", 0, 100, 50) / 100
# NOUVEAU : Option pour sauter des frames pour la fluidité
skip_frames = st.sidebar.slider("Saut d'images (Fluidité)", 1, 10, 5)

model = load_model(version_n)

st.title("🛡️ Analyse Industrielle Stabilisée")

# --- LOGIQUE VIDÉO STABILISÉE ---
v_file = st.file_uploader("Charger Vidéo", type=['mp4', 'avi'])

if v_file and model:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(v_file.read())
    vf = cv2.VideoCapture(tfile.name)
    
    # On crée un conteneur unique pour éviter que toute la page bouge
    container = st.empty() 
    btn_stop = st.button("Arrêter l'analyse")

    frame_count = 0
    last_label = "Analyse..."

    while vf.isOpened():
        ret, frame = vf.read()
        if not ret or btn_stop: break

        frame_count += 1
        
        # 🚀 OPTIMISATION : On n'analyse que 1 image sur X
        if frame_count % skip_frames == 0:
            cv2.imwrite("current_frame.jpg", frame)
            res = model.predict("current_frame.jpg").json()
            
            if res.get('predictions'):
                pred = res['predictions'][0]
                # On ne met à jour le label que s'il dépasse le seuil
                if pred.get('confidence', 0) >= threshold:
                    last_label = f"{pred.get('top', '???')} ({pred.get('confidence', 0):.1%})"
                else:
                    last_label = "Sous le seuil de confiance"

        # On dessine toujours le dernier label connu pour éviter que le texte disparaisse
        cv2.putText(frame, last_label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # Affichage fluide dans le conteneur dédié
        container.image(frame, channels="BGR", use_container_width=True)

    vf.release()
