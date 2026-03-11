import streamlit as st
from roboflow import Roboflow
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="IA Usinage - Version Finale", layout="wide", page_icon="⚙️")

API_KEY = "9FcisW7nvl380crhBt6e"
PROJECT_ID = "usinage-1uqck"
VERSION = 2  

@st.cache_resource
def load_model():
    try:
        rf = Roboflow(api_key=API_KEY)
        project = rf.workspace().project(PROJECT_ID)
        return project.version(VERSION).model
    except Exception as e:
        st.error(f"Erreur de chargement du modèle : {e}")
        return None

st.sidebar.title("🛠️ Menu de Contrôle")
mode = st.sidebar.selectbox("Mode d'analyse", ["Image Unique", "Dossier d'Images", "Vidéo"])

st.sidebar.divider()
st.sidebar.subheader("Réglages")
threshold = st.sidebar.slider("Seuil de Confiance (%)", 0, 100, 50) / 100

model = load_model()

st.title("⚙️ Classification Industrielle")
st.info(f"Modèle : `{PROJECT_ID}` | Version : `{VERSION}`")

if model:
    if mode == "Image Unique":
        file = st.file_uploader("Charger une image", type=['jpg', 'jpeg', 'png'])
        if file:
            img = Image.open(file)
            col1, col2 = st.columns(2)
            col1.image(img, caption="Pièce à analyser", use_container_width=True)
            
            if col2.button("🔍 Lancer l'Analyse"):
                temp_p = "predict_temp.jpg"
                img.save(temp_p)
                
                res = model.predict(temp_p).json()
                preds = res.get('predictions', [])
                
                if preds:
                    top_data = preds[0]
                    classe = top_data.get('top', 'Inconnu')
                    conf = top_data.get('confidence', 0)
                    
                    if conf >= threshold:
                        col2.success(f"### Résultat : **{classe}**")
                        col2.metric("Confiance", f"{conf:.2%}")
                    else:
                        col2.warning(f"⚠️ Confiance trop faible : {classe} ({conf:.1%})")
                    
                    st.divider()
                    with st.expander("📄 Voir le fichier JSON complet"):
                        st.json(res)

    elif mode == "Dossier d'Images":
        st.header("📁 Analyse par lot")
        files = st.file_uploader("Sélectionner des images", accept_multiple_files=True, type=['jpg', 'png', 'jpeg'])
        
        if files and st.button("Analyser le dossier"):
            grid = st.columns(3)
            for i, f in enumerate(files):
                img = Image.open(f)
                temp_p = f"batch_{i}.jpg"
                img.save(temp_p)
                
                data = model.predict(temp_p).json()
                preds = data.get('predictions', [])
                
                if preds:
                    top = preds[0]
                    label = top.get('top', '???') if top.get('confidence', 0) >= threshold else "❌ Incertain"
                    grid[i % 3].image(img, caption=f"{f.name} : {label}", use_container_width=True)

    elif mode == "Vidéo":
        st.header("🎥 Analyse Vidéo Stabilisée")
        v_file = st.file_uploader("Charger une vidéo", type=['mp4', 'avi', 'mov'])
        
        if v_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(v_file.read())
            vf = cv2.VideoCapture(tfile.name)
            
            video_placeholder = st.empty()
            btn_stop = st.button("Arrêter l'analyse")

            last_label = "Initialisation..."
            frame_count = 0

            while vf.isOpened():
                ret, frame = vf.read()
                if not ret or btn_stop: break
                
                frame_count += 1
                
                if frame_count % 5 == 0:
                    cv2.imwrite("frame_temp.jpg", frame)
                    data = model.predict("frame_temp.jpg").json()
                    preds = data.get('predictions', [])
                    
                    if preds:
                        top = preds[0]
                        if top.get('confidence', 0) >= threshold:
                            last_label = f"{top.get('top')} ({top.get('confidence'):.1%})"
                        else:
                            last_label = "Sous le seuil de confiance"

                cv2.putText(frame, last_label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                
                video_placeholder.image(frame, channels="BGR", use_container_width=True)
            
            vf.release()
else:
    st.error("❌ Le modèle Version 2 n'a pas pu être chargé. Vérifiez l'entraînement sur Roboflow.")
