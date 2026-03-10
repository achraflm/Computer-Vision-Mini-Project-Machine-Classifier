import streamlit as st
from roboflow import Roboflow
import numpy as np
from PIL import Image
import os

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Usinage Detection", layout="wide")
st.title("🛠️ Usinage Detection Dashboard")

# Utilisation des Secrets Streamlit (recommandé) ou Sidebar
ROBOFLOW_API_KEY = st.sidebar.text_input("Enter Roboflow API KEY", type="password")
PROJECT_ID = "usinage-1uqck" 
MODEL_VERSION = 1 

# --- 2. CHARGEMENT DU MODÈLE (SÉCURISÉ) ---
model = None

if ROBOFLOW_API_KEY:
    try:
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        # On spécifie le workspace si nécessaire, sinon Roboflow prend le défaut
        project = rf.workspace().project(PROJECT_ID)
        model = project.version(MODEL_VERSION).model
        st.sidebar.success("✅ Modèle chargé avec succès !")
    except Exception as e:
        st.sidebar.error(f"❌ Erreur de connexion : {e}")
else:
    st.warning("Veuillez entrer votre clé API dans la barre latérale.")

# --- 3. INTERFACE ---
uploaded_file = st.file_uploader("Upload a machining part image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Image Originale", use_container_width=True)
    
    with col2:
        if st.button("🔍 Run Detection"):
            if model is None:
                st.error("Le modèle n'est pas prêt. Vérifiez votre clé API.")
            else:
                with st.spinner('Analyse en cours...'):
                    # FIX: On simplifie l'appel pour éviter les erreurs d'arguments
                    # On enlève 'overlap' qui peut varier selon le type de modèle
                    prediction_res = model.predict(img_array, confidence=40)
                    
                    # 4. AFFICHAGE DES RÉSULTATS
                    results_json = prediction_res.json()
                    predictions = results_json.get('predictions', [])
                    
                    if not predictions:
                        st.info("Aucun objet détecté.")
                    else:
                        st.success(f"{len(predictions)} objets détectés !")
                        # Sauvegarde et affichage
                        prediction_res.save("prediction.jpg")
                        st.image("prediction.jpg", caption="Résultats", use_container_width=True)
                        
                        with st.expander("Voir les données JSON"):
                            st.json(results_json)
