import streamlit as st
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import unicodedata
from difflib import get_close_matches

# ─── Chargement et inférence des colonnes de la base tournée ─────────────────────

@st.cache_data(show_spinner=False)
def load_base():
    df = pd.read_excel("Base_tournees_KML_coordonnees.xlsx")
    lower = [col.lower() for col in df.columns]

    def find_col(key):
        """Retourne le nom de colonne le plus proche de key, ou None."""
        matches = get_close_matches(key, lower, n=1, cutoff=0.6)
        return matches[0] if matches else None

    # on cherche 3 colonnes dans df : tournée, latitude, longitude
    col_tour = find_col("tour")
    col_lat   = find_col("lat")
    col_lon   = find_col("lon")

    # si l’une manque, on stoppe
    missing = [k for k, c in [("tournée", col_tour), ("latitude", col_lat), ("longitude", col_lon)] if c is None]
    if missing:
        st.error(f"Colonnes manquantes dans la base tournée : {', '.join(missing)}.")
        st.stop()

    return df, col_tour, col_lat, col_lon

# ─── Fonction de recherche de la tournée la plus proche ──────────────────────────

def retrieve_tournee(base_df, col_tour, col_lat, col_lon, latitude, longitude):
    # calcul des distances
    base_df = base_df.copy()
    base_df["__dist"] = base_df.apply(
        lambda r: geodesic((latitude, longitude), (r[col_lat], r[col_lon])).meters,
        axis=1
    )
    # on retourne la ligne qui minimise __dist
    row = base_df.loc[ base_df["__dist"].idxmin() ]
    return row[col_tour], row["__dist"]

# ─── Interface Streamlit ────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="🛻 Attribution automatisée de tournées", layout="wide")
    st.markdown(
        """
        <style>
        body { background-color: #f0f2f6; }
        h1 { color: #0d3b66; }
        footer { visibility: hidden; }
        #made-by { text-align: center; padding: 10px; font-size:12px; color:#999; }
        </style>
        """, unsafe_allow_html=True
    )

    st.title("🛻 Attribution automatique de tournées")
    st.markdown("Téléversez un fichier Excel contenant les colonnes **adresse**, **code postal**, et **ville**.")

    # upload du fichier client
    uploaded_file = st.file_uploader("Téléversez un fichier Excel", type=["xlsx"])
    if not uploaded_file:
        return

    df_input = pd.read_excel(uploaded_file)
    cols_lower = [c.lower() for c in df_input.columns]

    # même logique : chercher adresse / code postal / ville
    def find_input_col(key):
        m = get_close_matches(key, cols_lower, n=1, cutoff=0.6)
        return m[0] if m else None

    col_addr = find_input_col("adresse")
    col_cp   = find_input_col("code postal")
    col_ville= find_input_col("ville")

    missing_in = [k for k, c in [("adresse", col_addr), ("code postal", col_cp), ("ville", col_ville)] if c is None]
    if missing_in:
        st.error(f"Le fichier doit contenir les colonnes : {', '.join(missing_in)}.")
        return

    # chargement de la base tournée
    base_df, col_tour, col_lat, col_lon = load_base()

    # on boucle sur chaque ligne
    results = []
    geolocator = Nominatim(user_agent="tournees-app")
    for idx, row in st.experimental_data_editor(df_input, num_rows="dynamic").iterrows():
        adr = f"{row[col_addr]}, {row[col_cp]}, {row[col_ville]}"
        try:
            loc = geolocator.geocode(adr, timeout=10)
            if not loc:
                raise ValueError("Introuvable")
            tour, dist = retrieve_tournee(base_df, col_tour, col_lat, col_lon, loc.latitude, loc.longitude)
        except Exception:
            tour, dist = "❌", np.nan
        results.append({"Tournee": tour, "Distance (m)": dist})

    df_out = df_input.join(pd.DataFrame(results))
    st.download_button("⬇️ Télécharger le résultat", df_out.to_excel(index=False), "resultat.xlsx")

    st.markdown('<div id="made-by">Made by Delestret Kim</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
