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

# --- 2. BARRE LATÉRALE ---
st.sidebar.title("🛠️ Configuration")
mode = st.sidebar.selectbox("Mode", ["Image Unique", "Dossier d'Images", "Vidéo"])
version_n = st.sidebar.number_input("Version", min_value=1, value=2)

st.sidebar.divider()
st.sidebar.subheader("Sensibilité")
# Le slider pour régler le seuil
threshold = st.sidebar.slider("Seuil de Confiance (%)", 0, 100, 50) / 100

model = load_model(version_n)

# --- 3. INTERFACE PRINCIPALE ---
st.title("⚙️ Système de Classification de Pièces")

if model:
    if mode == "Image Unique":
        st.header("📸 Analyse d'une pièce")
        file = st.file_uploader("Charger une image", type=['jpg', 'jpeg', 'png'])
        
        if file:
            img = Image.open(file)
            col1, col2 = st.columns(2)
            col1.image(img, caption="Pièce à analyser", use_container_width=True)
            
            if col2.button("🔍 Lancer l'Analyse"):
                temp_p = "temp_predict.jpg"
                img.save(temp_p)
                
                # --- RÉCUPÉRATION ET PARSING DU JSON ---
                res = model.predict(temp_p).json()
                
                # Extraction selon ta structure JSON précise
                try:
                    # On va chercher directement dans le premier élément de 'predictions'
                    prediction_data = res['predictions'][0]
                    classe_detectee = prediction_data['top']
                    confiance = prediction_data['confidence']

                    # --- LOGIQUE DE SEUIL ---
                    if confiance >= threshold:
                        col2.success(f"### Résultat : **{classe_detectee}**")
                        col2.metric("Niveau de certitude", f"{confiance:.2%}")
                    else:
                        col2.warning(f"⚠️ Détection incertaine : {classe_detectee} ({confiance:.2%})")
                        col2.info(f"Le seuil est actuellement réglé à {threshold*100:.0f}%")
                
                except (KeyError, IndexError):
                    col2.error("Erreur de lecture des données JSON.")

                # Affichage des données brutes en dessous
                with st.expander("📄 Voir le fichier JSON complet"):
                    st.json(res)

    elif mode == "Dossier d'Images":
        st.header("📁 Analyse par lot")
        files = st.file_uploader("Upload Dossier", accept_multiple_files=True)
        if files and st.button("Analyser le lot"):
            grid = st.columns(3)
            for i, f in enumerate(files):
                img = Image.open(f)
                temp_p = f"batch_{i}.jpg"
                img.save(temp_p)
                
                data = model.predict(temp_p).json()
                try:
                    pred = data['predictions'][0]
                    # On affiche la classe seulement si > seuil
                    label = pred['top'] if pred['confidence'] >= threshold else "❌ Inconnu"
                    grid[i % 3].image(img, caption=f"{f.name} : {label}")
                except:
                    grid[i % 3].image(img, caption=f"{f.name} : Erreur")

    elif mode == "Vidéo":
        st.header("🎥 Analyse Vidéo")
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
                    data = model.predict("frame.jpg").json()
                    
                    try:
                        pred = data['predictions'][0]
                        if pred['confidence'] >= threshold:
                            text = f"{pred['top']} ({pred['confidence']:.1%})"
                            cv2.putText(frame, text, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    except: pass
                    
                    st_frame.image(frame, channels="BGR", use_container_width=True)
                vf.release()
