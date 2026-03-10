import streamlit as st
from roboflow import Roboflow
from PIL import Image
import numpy as np

# --- 1. CONFIGURATION ---
API_KEY = "9FcisW7nvl380crhBt6e"
PROJECT_ID = "usinage-1uqck" 

st.set_page_config(page_title="Usinage Classifier V2", page_icon="⚙️")
st.title("🔧 Usinage Part Classifier")

# --- 2. MENU DE CONFIGURATION DANS LA SIDEBAR ---
st.sidebar.header("Paramètres")
# Ici, tu peux changer entre 1, 2, ou 3 si tu fais de nouveaux tests
version_number = st.sidebar.number_input("Version du modèle", min_value=1, value=2, step=1)

@st.cache_resource
def load_model(v):
    try:
        rf = Roboflow(api_key=API_KEY)
        project = rf.workspace().project(PROJECT_ID)
        # On charge la version choisie (ici V2 par défaut)
        model = project.version(v).model
        return model
    except Exception as e:
        st.sidebar.error(f"Erreur technique : {e}")
        return None

# Chargement du modèle avec la version choisie
model = load_model(version_number)

# --- 3. INTERFACE UTILISATEUR ---
if model:
    st.sidebar.success(f"✅ Modèle V{version_number} chargé !")
else:
    st.sidebar.warning(f"⚠️ Modèle V{version_number} non trouvé.")
    st.info("Vérifiez sur Roboflow que la version est bien 'Trained'.")

uploaded_file = st.file_uploader("Charger une image de pièce mécanique...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image à analyser", use_container_width=True)
    
    if st.button("🔍 Identifier avec la V2"):
        if model:
            with st.spinner('Analyse en cours...'):
                # Rappel : C'est de la CLASSIFICATION (pas de boîtes)
                prediction = model.predict(np.array(image)).json()
                
                if "predictions" in prediction and len(prediction["predictions"]) > 0:
                    top = prediction["predictions"][0]
                    st.divider()
                    st.header(f"Résultat : :blue[{top['class']}]")
                    st.write(f"Niveau de confiance : **{top['confidence']:.2%}**")
                else:
                    st.warning("Aucun résultat pour cette image.")
        else:
            st.error("Impossible de lancer l'analyse : modèle non chargé.")
