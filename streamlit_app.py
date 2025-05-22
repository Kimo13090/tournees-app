import streamlit as st
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
import unicodedata
import difflib

# --- Style CSS
st.markdown("""
<style>
body { background-color: #f0f2f6; }
h1 { color: #0d3b66; }
footer { visibility: hidden; }
#made-by { text-align: center; padding: 10px; font-size:12px; color:#999; }
</style>
""", unsafe_allow_html=True)

# Titre
st.title("🚚 Attribution automatique de tournées")
st.markdown("---")

# Chargement des données
@st.cache_data
def load_base():
    df = pd.read_excel("Base_tournees_KML_coordonnees.xlsx")
    # normaliser colonnes
    df.rename(columns={c: unicodedata.normalize('NFKD', c).encode('ascii','ignore').decode().lower() for c in df.columns}, inplace=True)
    # préparer arrays
    lat_arr = np.radians(df['latitude'].values)
    lon_arr = np.radians(df['longitude'].values)
    tourn_arr = df['tournee'].values
    return df, lat_arr, lon_arr, tourn_arr

# Détection colonnes utilisateur
def detect_cols(cols):
    norm = [unicodedata.normalize('NFKD', c).encode('ascii','ignore').decode().lower().replace(' ', '') for c in cols]
    mapping = {}
    for target in ['adresse','codepostal','ville']:
        match = difflib.get_close_matches(target, norm, n=1, cutoff=0.5)
        if match:
            mapping[target] = cols[norm.index(match[0])]
    return mapping

# Haversine vectorisé
def vector_haversine(lat, lon, lat_arr, lon_arr):
    R = 6371.0
    dlat = lat_arr - np.radians(lat)
    dlon = lon_arr - np.radians(lon)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat))*np.cos(lat_arr)*np.sin(dlon/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

# App UI
st.markdown("Téléversez un fichier Excel contenant des colonnes d'adresse, code postal, ville (n'importe l'ordre).")
uploaded = st.file_uploader("Choisir un fichier Excel", type=['xlsx'])

if uploaded:
    df_in = pd.read_excel(uploaded)
    cols = detect_cols(df_in.columns)
    if len(cols) < 3:
        st.error("Impossible de détecter adresse, code postal et ville automatiquement. Vérifiez votre fichier.")
    else:
        base_df, lat_arr, lon_arr, tourn_arr = load_base()
        geolocator = Nominatim(user_agent='tournees-app')
        results = []
        for i, row in df_in.iterrows():
            adr = f"{row[cols['adresse']]}, {row[cols['codepostal']]}, {row[cols['ville']]}"
            loc = geolocator.geocode(adr)
            if loc:
                dist_arr = vector_haversine(loc.latitude, loc.longitude, lat_arr, lon_arr)
                idx = np.argmin(dist_arr)
                results.append((loc.latitude, loc.longitude, tourn_arr[idx], round(dist_arr[idx], 3)))
            else:
                results.append((None, None, 'Non trouvée', None))
        df_in[['latitude','longitude','tournée','distance_km']] = pd.DataFrame(results, index=df_in.index)
        st.success("Traitement terminé !")
        st.dataframe(df_in)
        # Téléchargement
        st.download_button(
            label='📥 Télécharger le résultat',
            data=df_in.to_excel(index=False, engine='openpyxl'),
            file_name='resultat_tournees.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

# Pied de page
st.markdown("<div id='made-by'>Made by Delestret Kim</div>", unsafe_allow_html=True)
