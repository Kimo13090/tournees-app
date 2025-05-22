
import streamlit as st
import pandas as pd
import numpy as np
import pgeocode
from sklearn.neighbors import BallTree
from rapidfuzz import fuzz
import unicodedata
import difflib

# Chargement de la base de tournées
@st.cache_data
def load_base():
    base = pd.read_excel("Base_tournees_KML_coordonnees.xlsx")
    base["coord"] = np.deg2rad(base[["Latitude", "Longitude"]].values)
    tree = BallTree(base["coord"].tolist(), metric="haversine")
    return base, tree

# Tentative de détection intelligente des colonnes
def detect_columns(df):
    targets = {"adresse": ["adresse", "rue", "address"],
               "code postal": ["cp", "code", "postal"],
               "ville": ["ville", "commune", "localite"]}
    mapping = {}
    for target, keys in targets.items():
        for col in df.columns:
            col_clean = unicodedata.normalize("NFKD", col.lower()).encode("ASCII", "ignore").decode()
            if any(fuzz.partial_ratio(col_clean, key) > 80 for key in keys):
                mapping[target] = col
                break
    return mapping

# Récupération des coordonnées centrées sur le code postal
def get_postal_centroid(cp_series):
    nomi = pgeocode.Nominatim("fr")
    lat = nomi.query_postal_code(cp_series.astype(str)).latitude.fillna(0).values
    lon = nomi.query_postal_code(cp_series.astype(str)).longitude.fillna(0).values
    return np.deg2rad(np.vstack([lat, lon]).T)

# Attribution de tournée
def assign_tournees(df_input, base_df, tree):
    pts = get_postal_centroid(df_input["code postal"])
    dist, ind = tree.query(pts, k=1)
    df_input["Tournee"] = base_df.iloc[ind[:,0]]["Tournee"].values
    df_input["Distance (km)"] = (dist[:,0] * 6371).round(2)
    return df_input

# --- UI ---
st.markdown("""
    <style>
    body { background-color: #f0f2f6; }
    h1 { color: #0d3b66; }
    footer {visibility: hidden;}
    #made-by { text-align: center; padding: 10px; font-size:12px; color:#999; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚚 Attribution automatique de tournées")
st.markdown("Téléversez un fichier Excel contenant les colonnes **adresse**, **code postal**, et **ville**.

L'application tentera de retrouver automatiquement la tournée correspondante pour chaque ligne.")

uploaded_file = st.file_uploader("Téléversez un fichier Excel", type=["xlsx"])

if uploaded_file:
    df_input = pd.read_excel(uploaded_file)
    col_map = detect_columns(df_input)

    if len(col_map) < 3:
        st.error("Le fichier doit contenir les colonnes : adresse, code postal, ville (ou noms similaires).")
    else:
        df_input.rename(columns=col_map, inplace=True)
        base_df, tree = load_base()
        df_out = assign_tournees(df_input, base_df, tree)
        st.success("Tournées attribuées avec succès ✅")
        st.dataframe(df_out)

        st.download_button("📥 Télécharger le fichier avec tournées",
                           data=df_out.to_excel(index=False),
                           file_name="resultat_tournees.xlsx")

st.markdown('<div id="made-by">Made by Delestret Kim</div>', unsafe_allow_html=True)
