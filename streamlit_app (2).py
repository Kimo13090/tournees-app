
import streamlit as st
import pandas as pd
import numpy as np
import os
from difflib import get_close_matches
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from unidecode import unidecode

st.set_page_config(page_title="Attribution de tournées", layout="centered")

def detect_column_name(possibles, colonnes):
    matches = get_close_matches(possibles.lower(), [c.lower() for c in colonnes], n=1, cutoff=0.5)
    if matches:
        for c in colonnes:
            if c.lower() == matches[0]:
                return c
    return None

@st.cache_data
def load_base():
    base_df = pd.read_excel("Base_tournees_KML_coordonnees.xlsx")
    lower = base_df.columns

    col_lat = detect_column_name("latitude", lower)
    col_lon = detect_column_name("longitude", lower)
    col_tour = detect_column_name("tournee", lower)

    if not all([col_lat, col_lon, col_tour]):
        missing = []
        if not col_lat: missing.append("latitude")
        if not col_lon: missing.append("longitude")
        if not col_tour: missing.append("tournée")
        st.error(f"Colonnes manquantes dans la base tournée : {', '.join(missing)}")
        return None, None, None, None

    return base_df, col_tour, col_lat, col_lon

def retrieve_tournee(base_df, col_tour, col_lat, col_lon, lat, lon):
    base_df = base_df.copy()
    base_df["distance"] = base_df.apply(lambda row: geodesic((lat, lon), (row[col_lat], row[col_lon])).meters, axis=1)
    return base_df.sort_values("distance").iloc[0][col_tour]

@st.cache_data
def load_cache():
    cache_file = "geocache.csv"
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file)
    else:
        return pd.DataFrame(columns=["adresse", "code_postal", "ville", "latitude", "longitude"])

def update_cache(cache_df, adresse, cp, ville, lat, lon):
    new_row = pd.DataFrame([[adresse, cp, ville, lat, lon]], columns=cache_df.columns)
    updated = pd.concat([cache_df, new_row], ignore_index=True)
    updated.to_csv("geocache.csv", index=False)
    return updated

# Interface utilisateur
st.markdown("""<style>
body { background-color: #f0f2f6; }
footer { visibility: hidden; }
#made-by { text-align: center; padding: 10px; font-size:12px; color:#999; }
</style>""", unsafe_allow_html=True)

st.title("🚚 Attribution automatique de tournées")
st.markdown("Téléversez un fichier Excel contenant les colonnes **adresse**, **code postal**, et **ville**.\nL'application trouvera la tournée la plus proche pour chaque ligne.")

uploaded = st.file_uploader("Téléversez un fichier Excel", type=["xlsx"])

if uploaded:
    df = pd.read_excel(uploaded)
    lower = df.columns

    col_adresse = detect_column_name("adresse", lower)
    col_cp = detect_column_name("code postal", lower)
    col_ville = detect_column_name("ville", lower)

    if not all([col_adresse, col_cp, col_ville]):
        st.error("Le fichier doit contenir les colonnes : adresse, code postal, ville (ou noms similaires).")
    else:
        base_df, col_tour, col_lat, col_lon = load_base()
        if base_df is not None:
            cache_df = load_cache()
            geolocator = Nominatim(user_agent="tournees-app")
            tournees = []

            for _, row in df.iterrows():
                try:
                    adr = str(row[col_adresse]).strip()
                    cp = str(row[col_cp]).strip()
                    vil = str(row[col_ville]).strip()

                    match = cache_df[(cache_df["adresse"] == adr) & (cache_df["code_postal"] == cp) & (cache_df["ville"] == vil)]

                    if not match.empty:
                        lat = match.iloc[0]["latitude"]
                        lon = match.iloc[0]["longitude"]
                    else:
                        loc = geolocator.geocode(f"{adr}, {cp}, {vil}")
                        if loc:
                            lat, lon = loc.latitude, loc.longitude
                            cache_df = update_cache(cache_df, adr, cp, vil, lat, lon)
                        else:
                            tournees.append("Adresse introuvable")
                            continue

                    tour = retrieve_tournee(base_df, col_tour, col_lat, col_lon, lat, lon)
                    tournees.append(tour)
                except:
                    tournees.append("Erreur")

            df["Tournée"] = tournees
            st.success("Traitement terminé ! Voici un aperçu :")
            st.dataframe(df)
            st.download_button("📥 Télécharger le fichier avec tournées", df.to_excel(index=False), file_name="resultat_tournees.xlsx")
