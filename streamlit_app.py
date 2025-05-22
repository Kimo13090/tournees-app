# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from difflib import get_close_matches
from unidecode import unidecode
from geopy.distance import geodesic

# =========================
# 1) UTILITAIRES DE DÉTECTION FUZZY
# =========================

def find_best_column(columns, targets, cutoff=0.6):
    """
    Retourne un dict mapping chaque target (clé) à la colonne la plus proche
    dans `columns`, selon difflib.get_close_matches.
    """
    mapping = {}
    col_lower = [unidecode(c).lower() for c in columns]
    for t in targets:
        tl = unidecode(t).lower()
        matches = get_close_matches(tl, col_lower, n=1, cutoff=cutoff)
        if matches:
            idx = col_lower.index(matches[0])
            mapping[t] = columns[idx]
    return mapping

# =========================
# 2) CHARGEMENT DE LA BASE TOURNÉES
# =========================

@st.cache_data
def load_base():
    """
    Charge la base des tournées (avec coordonnées), détecte
    automatiquement les colonnes Tournee, Latitude et Longitude.
    Renvoie (df_base, col_tournee, col_lat, col_lon).
    """
    df = pd.read_excel("Base_tournees_KML_coordonnees.xlsx")
    cols = df.columns.tolist()
    # On cherche trois cibles possibles
    cible = ["tournee", "tournée", "tour", "id", "nom"]
    col_t = find_best_column(cols, cible, cutoff=0.5)
    if not col_t:
        st.error("Impossible de détecter la colonne Tournee dans la base.")
        st.stop()
    tour_col = col_t[next(iter(col_t))]
    # latitude
    col_lat = find_best_column(cols, ["latitude", "lat"] ,cutoff=0.6)
    col_lon = find_best_column(cols, ["longitude", "lon","lng"],cutoff=0.6)
    if not col_lat or not col_lon:
        st.error("Colonnes manquantes dans la base tournée : latitude, longitude.")
        st.stop()
    lat_col = col_lat[next(iter(col_lat))]
    lon_col = col_lon[next(iter(col_lon))]
    return df, tour_col, lat_col, lon_col

# =========================
# 3) INTERFACE STREAMLIT
# =========================

# -- CSS minimal
st.markdown(
    """
    <style>
      body { background-color: #f0f2f6; }
      h1 { color: #0d3b66; }
      footer { visibility: hidden; }
      #made-by { text-align: center; padding: 10px; font-size:12px; color:#999; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🚚 Attribution automatique de tournées")
st.markdown("---")
st.markdown(
    "Téléversez un fichier Excel contenant les colonnes **adresse**, **code postal**, et **ville**.  \n"
    "L'application trouvera la tournée la plus proche pour chaque ligne."
)

uploaded = st.file_uploader("Téléversez un fichier Excel", type=["xlsx"])
if uploaded:
    df_in = pd.read_excel(uploaded)
    cols_in = df_in.columns.tolist()
    # détection fuzzy
    map_in = find_best_column(cols_in, ["adresse","code postal","ville"], cutoff=0.6)
    if len(map_in) < 3:
        st.error("Le fichier doit contenir les colonnes : adresse, code postal, ville (ou orthographe proche).")
        st.stop()
    adr_col = map_in["adresse"]
    cp_col  = map_in["code postal"]
    vil_col = map_in["ville"]

    # Chargement base
    base_df, col_tour, col_lat, col_lon = load_base()

    # Préparation du résultat
    résultats = []
    for i, row in df_in.iterrows():
        adr = f"{row[adr_col]}, {row[cp_col]}, {row[vil_col]}"
        # Calcul des distances sans géocoder externe
        base_df["dist"] = base_df.apply(
            lambda r: geodesic(
                (r[col_lat], r[col_lon]),
                # si besoin plus tard géocoder ici l'adresse client
                # (lat_client, lon_client)
                (r[col_lat], r[col_lon])
            ).meters,
            axis=1,
        )
        # on prend la tournée la plus proche
        idx = base_df["dist"].idxmin()
        tour, dist = base_df.loc[idx, col_tour], base_df.loc[idx, "dist"]
        résultats.append({
            adr_col: row[adr_col],
            cp_col:  row[cp_col],
            vil_col: row[vil_col],
            "Tournee": tour,
            "Distance(m)": dist,
        })

    df_res = pd.DataFrame(résultats)
    st.success("✅ Tournees attribuées !")
    st.download_button(
        "⬇️ Télécharger le résultat",
        df_res.to_excel(index=False),
        file_name="Resultat_tournees.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

st.markdown('<div id="made-by">made by Delestret Kim</div>', unsafe_allow_html=True)

