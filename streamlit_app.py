import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim
from scipy.spatial import cKDTree
import numpy as np

# Chargement des bases
@st.cache_data
def load_data():
    base = pd.read_excel("Base_tournees_KML_coordonnees.xlsx")
    # on s'assure d'avoir 'latitude', 'longitude', 'Tournee'
    return base

# Construction du KD-Tree
@st.cache_data
def build_tree(base):
    coords = np.vstack((base['latitude'], base['longitude'])).T
    tree = cKDTree(coords)
    return tree

# Géocodage avec cache
@st.cache_data
def geocode_address(adresse_complete):
    geolocator = Nominatim(user_agent="tournees-app")
    loc = geolocator.geocode(adresse_complete)
    if loc:
        return loc.latitude, loc.longitude
    return None, None

# Détection des colonnes similaires
import difflib

def detect_columns(cols):
    targets = ['adresse', 'code postal', 'ville']
    found = {}
    for target in targets:
        match = difflib.get_close_matches(target, [c.lower() for c in cols], n=1, cutoff=0.6)
        if match:
            original = [c for c in cols if c.lower() == match[0]][0]
            found[target] = original
    return found

# Interface Streamlit
st.title("🚚 Attribution automatique de tournées")
st.markdown("""
Téléversez un fichier Excel contenant les colonnes **adresse**, **code postal**, et **ville**
(peu importe l'ordre ou les majuscules). L'application attribue automatiquement la tournée la plus proche.
""")

uploaded_file = st.file_uploader("Téléversez un fichier Excel", type=["xlsx"])
if uploaded_file:
    df_input = pd.read_excel(uploaded_file)
    col_map = detect_columns(df_input.columns)
    if len(col_map) < 3:
        st.error("Le fichier doit contenir les colonnes : adresse, code postal, ville (ou noms similaires).")
    else:
        base = load_data()
        tree = build_tree(base)
        results = []
        for _, row in df_input.iterrows():
            adresse_complete = f"{row[col_map['adresse']]}, {row[col_map['code postal']]}, {row[col_map['ville']]}"
            lat, lon = geocode_address(adresse_complete)
            if lat is None:
                results.append((None, None, 'Non trouvée', None))
            else:
                # recherche du plus proche voisin
                dist, idx = tree.query((lat, lon))
                tournee = base.iloc[idx]['Tournee']
                results.append((lat, lon, tournee, dist))
        # ajout des colonnes résultats
        df_input[['latitude','longitude','tournee','distance_km']] = pd.DataFrame(results, index=df_input.index)
        st.success("Tournées attribuées avec succès !")
        st.dataframe(df_input)
        # export
        towrite = pd.ExcelWriter('resultat_tournees.xlsx', engine='openpyxl')
        df_input.to_excel(towrite, index=False)
        towrite.save()
        st.download_button("📥 Télécharger le fichier avec tournées", data=open('resultat_tournees.xlsx','rb'), file_name='resultat_tournees.xlsx')

