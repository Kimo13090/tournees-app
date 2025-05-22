import streamlit as st
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from difflib import get_close_matches
from unidecode import unidecode

# --- Style CSS ---
st.markdown("""
<style>
body { background-color: #f0f2f6; }
h1 { color: #0d3b66; }
footer { visibility: hidden; }
#made-by { text-align: center; padding: 10px; font-size:12px; color:#999; }
</style>
""", unsafe_allow_html=True)

# --- Fonctions utilitaires ---
@st.cache_data
def geocode_address(geolocator, adresse_complete):
    try:
        return geolocator.geocode(adresse_complete)
    except:
        return None

@st.cache_data
def load_base():
    df = pd.read_excel("Base_tournees_KML_coordonnees.xlsx")
    df.columns = [unidecode(col.lower().strip()) for col in df.columns]

    lower = df.columns
    col_lat = lower[get_close_matches("lat", lower, n=1, cutoff=0.6)][0]
    col_lon = lower[get_close_matches("lon", lower, n=1, cutoff=0.6)][0]
    col_tour = lower[get_close_matches("tournee", lower, n=1, cutoff=0.6)][0]

    return df, col_tour, col_lat, col_lon

def detect_columns(df):
    cols = df.columns
    mapping = {}
    for col in cols:
        name = unidecode(col.lower().strip())
        if "adresse" in name: mapping["adresse"] = col
        elif "postal" in name: mapping["code postal"] = col
        elif "ville" in name: mapping["ville"] = col
    return mapping

def retrieve_tournee(base_df, lat, lon):
    distances = base_df.apply(
        lambda r: geodesic((lat, lon), (r["lat"], r["lon"])).meters, axis=1
    )
    min_index = distances.idxmin()
    return base_df.loc[min_index, "tournee"], distances[min_index]

# --- Interface Streamlit ---
st.title("🚚 Attribution automatique de tournées")
st.markdown("""
Téléversez un fichier Excel contenant les colonnes **adresse**, **code postal**, et **ville**.  
L'application trouvera la tournée la plus proche pour chaque ligne.
""")

uploaded_file = st.file_uploader("Téléversez un fichier Excel", type=["xlsx"])

if uploaded_file:
    df_input = pd.read_excel(uploaded_file)
    df_input = df_input.head(10)  # Limiter à 10 lignes pour test rapide
    col_map = detect_columns(df_input)

    if len(col_map) < 3:
        st.error("Le fichier doit contenir les colonnes : adresse, code postal, ville")
    else:
        base_df, col_tour, col_lat, col_lon = load_base()

        geolocator = Nominatim(user_agent="tournees-app")
        coordonnees = []

        for idx, row in df_input.iterrows():
            adresse_complete = f"{row[col_map['adresse']]}, {row[col_map['code postal']]}, {row[col_map['ville']]}"
            with st.spinner(f"Géocodage ligne {idx+1}/{len(df_input)}..."):
                loc = geocode_address(geolocator, adresse_complete)
                if loc:
                    tournee, distance = retrieve_tournee(base_df, loc.latitude, loc.longitude)
                    coordonnees.append({"Tournee": tournee, "Distance (m)": round(distance, 1)})
                else:
                    coordonnees.append({"Tournee": "Non trouvée", "Distance (m)": None})

        df_resultat = pd.concat([df_input.reset_index(drop=True), pd.DataFrame(coordonnees)], axis=1)
        st.success("Traitement terminé")
        st.dataframe(df_resultat)

        st.download_button("🔳 Télécharger les résultats", df_resultat.to_csv(index=False).encode('utf-8'), file_name="resultats_tournees.csv", mime="text/csv")

st.markdown("""
<div id="made-by">Made by <b>Delestret Kim</b></div>
""", unsafe_allow_html=True)
