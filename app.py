import streamlit as st
import pandas as pd
import requests
import time
from math import radians, sin, cos, sqrt, atan2

# --- Configuration ---
USER_AGENT = "TourneeLocator/1.0 (contact@votredomaine.com)"

# --- Fonctions utilitaires ---
def distance_haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Rayon de la Terre en km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

@st.cache_data
def load_tournees():
    # Charge la base de référence des tournées (avec Lat/Lon)
    return pd.read_excel("Base_tournees_KML_coordonnees.xlsx")

@st.cache_data
def geocode(address):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": address, "format": "json", "limit": 1}
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, params=params, headers=headers)
    if resp.status_code == 200 and resp.json():
        data = resp.json()[0]
        return float(data["lat"]), float(data["lon"])
    return None, None

# --- App principale ---
def main():
    st.title("Attribution Automatique des Tournées")
    st.write("Téléversez un fichier clients (Excel/CSV) et sélectionnez vos colonnes d'adresse, code postal et ville.")

    uploaded = st.file_uploader("Fichier Excel/CSV", type=["xlsx", "xls", "csv"])
    if not uploaded:
        st.info("En attente de fichier...")
        return

    # Lecture du fichier
    if uploaded.name.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded)
    else:
        df = pd.read_csv(uploaded)

    st.write("**Colonnes détectées :**", df.columns.tolist())

    # Sélection interactive des colonnes
    addr_cols = st.multiselect(
        "Colonnes d'adresse (plusieurs possibles)",
        df.columns.tolist(),
        default=[c for c in df.columns if any(k in c.lower() for k in ['voie','rue','chemin','addr','avenue','av','bd','lotissement','impasse','résidence','residence'])]
    )
    cp_col = st.selectbox("Colonne Code Postal", [""] + df.columns.tolist())
    ville_col = st.selectbox("Colonne Ville", [""] + df.columns.tolist())

    if not addr_cols:
        st.error("Veuillez sélectionner au moins une colonne d'adresse.")
        return

    # Construction de l'adresse complète
    full_addr = df[addr_cols].astype(str).agg(' '.join, axis=1)
    if cp_col:
        full_addr += ' ' + df[cp_col].astype(str)
    if ville_col:
        full_addr += ' ' + df[ville_col].astype(str)
    df['_full_address'] = full_addr

    # Géocodage
    lats, lons = [], []
    for address in df['_full_address']:
        lat, lon = geocode(address)
        lats.append(lat)
        lons.append(lon)
        time.sleep(1)  # Respect fair-use de Nominatim
    df['Latitude'] = lats
    df['Longitude'] = lons

    # Chargement des tournées
    df_tournees = load_tournees()

    # Attribution de la tournée la plus proche
    assigned = []
    for _, client in df.iterrows():
        best = (None, float('inf'))
        if pd.notna(client['Latitude']) and pd.notna(client['Longitude']):
            for _, tour in df_tournees.iterrows():
                d = distance_haversine(
                    client['Latitude'], client['Longitude'],
                    tour['Latitude'], tour['Longitude']
                )
                if d < best[1]:
                    best = (tour['Tournée'], d)
        assigned.append(best[0] if best[0] else 'Non trouvé')
    df['Tournée attribuée'] = assigned

    # Affichage et téléchargement
    st.dataframe(df)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Télécharger le fichier enrichi",
        csv,
        "clients_tournees_attribues.csv",
        "text/csv"
    )

if __name__ == '__main__':
    main()
