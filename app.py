import streamlit as st
from roboflow import Roboflow
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="IA Industrielle Unifiée", layout="wide", page_icon="⚙️")

# ❌ UN SEUL ID POUR TOUT
API_KEY = "9FcisW7nvl380crhBt6e"
PROJECT_ID = "test-pz2em-ghj7h" 
VERSION = 2

@st.cache_resource
def load_unified_model():
    try:
        rf = Roboflow(api_key=API_KEY)
        project = rf.workspace().project(PROJECT_ID)
        return project.version(VERSION).model
    except Exception as e:
        st.error(f"Erreur de connexion au projet {PROJECT_ID} : {e}")
        return None

# --- 2. BARRE LATÉRALE ---
st.sidebar.title("🎮 Contrôle Unique")
# L'utilisateur choisit comment interpréter les données du modèle
task = st.sidebar.radio("Mode d'affichage", ["Détection (Boîtes)", "Classification (Texte seul)"])
mode = st.sidebar.selectbox("Source", ["Image Unique", "Dossier d'Images", "Vidéo"])

st.sidebar.divider()
threshold = st.sidebar.slider("Seuil de Confiance (%)", 0, 100, 50)

model = load_unified_model()

# --- 3. INTERFACE PRINCIPALE ---
st.title(f"🛠️ Analyse Industrielle : {PROJECT_ID}")
st.caption(f"Utilisation du modèle de détection pour la {task.lower()}")

if model:
    # --- MODE IMAGE UNIQUE ---
    if mode == "Image Unique":
        file = st.file_uploader("Charger une image", type=['jpg', 'jpeg', 'png'])
        if file:
            img = Image.open(file)
            col1, col2 = st.columns(2)
            col1.image(img, caption="Source", use_container_width=True)
            
            if col2.button("🚀 Lancer l'Analyse"):
                temp_p = "unified_predict.jpg"
                img.save(temp_p)
                
                # Exécution de la prédiction
                prediction = model.predict(temp_p, confidence=threshold)
                
                if task == "Détection (Boîtes)":
                    prediction.save("res_detect.jpg")
                    col2.image("res_detect.jpg", caption="Résultat avec Bounding Boxes")
                else:
                    # On transforme la détection en classification
                    # On prend l'objet avec la plus haute confiance
                    preds = prediction.json().get('predictions', [])
                    if preds:
                        top = preds[0] # Le premier est souvent le plus confiant
                        col2.success(f"### Classe détectée : **{top['class']}**")
                        col2.metric("Confiance", f"{top['confidence']:.2%}")
                    else:
                        col2.warning("Aucun objet détecté au-dessus du seuil.")
                
                with st.expander("🔍 Voir les données JSON"):
                    st.json(prediction.json())

    # --- MODE DOSSIER ---
    elif mode == "Dossier d'Images":
        files = st.file_uploader("Upload Dossier", accept_multiple_files=True)
        if files and st.button("Analyser le lot"):
            grid = st.columns(3)
            for i, f in enumerate(files):
                img = Image.open(f)
                path = f"batch_{i}.jpg"
                img.save(path)
                
                prediction = model.predict(path, confidence=threshold)
                if task == "Détection (Boîtes)":
                    prediction.save(f"out_{i}.jpg")
                    grid[i % 3].image(f"out_{i}.jpg", caption=f.name)
                else:
                    preds = prediction.json().get('predictions', [])
                    label = preds[0]['class'] if preds else "Inconnu"
                    grid[i % 3].image(img, caption=f"{f.name} : {label}")

    # --- MODE VIDÉO ---
    elif mode == "Vidéo":
        v_file = st.file_uploader("Charger Vidéo", type=['mp4', 'avi'])
        if v_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(v_file.read())
            vf = cv2.VideoCapture(tfile.name)
            placeholder = st.empty()
            
            f_count = 0
            last_text = "Analyse..."

            if st.button("Démarrer la Vidéo"):
                while vf.isOpened():
                    ret, frame = vf.read()
                    if not ret: break
                    f_count += 1
                    
                    if f_count % 5 == 0:
                        cv2.imwrite("frame.jpg", frame)
                        prediction = model.predict("frame.jpg", confidence=threshold)
                        
                        if task == "Détection (Boîtes)":
                            prediction.save("frame_res.jpg")
                            frame = cv2.imread("frame_res.jpg")
                        else:
                            preds = prediction.json().get('predictions', [])
                            last_text = f"{preds[0]['class']} ({preds[0]['confidence']:.1%})" if preds else "..."
                    
                    if task == "Classification (Texte seul)":
                        cv2.putText(frame, last_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    
                    placeholder.image(frame, channels="BGR", use_container_width=True)
                vf.release()
