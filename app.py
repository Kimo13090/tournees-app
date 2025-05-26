import streamlit as st
import pandas as pd
import requests
import time
from math import radians, sin, cos, sqrt, atan2

# --- Configuration ---
USER_AGENT = "TourneeLocator/1.0 (contact@votredomaine.com)"

# --- Fonctions utilitaires ---
def distance_haversine(lat1, lon1, lat2, lon2):
    """Calcule la distance en km entre deux points GPS."""
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

@st.cache_data
def load_tournees():
    """Charge la base des tournées depuis un fichier Excel"""
    return pd.read_excel("Base_tournees_KML_coordonnees.xlsx")

@st.cache_data
def geocode(address):
    """Géocode une adresse via Nominatim OpenStreetMap"""
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": address, "format": "json", "limit": 1}
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, params=params, headers=headers)
    if resp.status_code == 200 and resp.json():
        data = resp.json()[0]
        return float(data["lat"]), float(data["lon"])
    return None, None

# --- Application Streamlit ---
def main():
    st.title("Attribution Automatique des Tournées")
    st.write("Upload un fichier clients (Adresse, Complément, Code postal, Ville) ; l'app géocode, calcule les distances et attribue la tournée ou 'HZ' si hors zone.")

    # Seuil de distance pour hors zone
    seuil_km = st.slider("Distance maximale (km) pour attribution de tournée (au-delà = HZ)", min_value=0.1, max_value=20.0, value=5.0, step=0.1)

    uploaded = st.file_uploader("Fichier Excel/CSV", type=["xlsx", "xls", "csv"])
    if not uploaded:
        st.stop()

    # Lecture du fichier client
    if uploaded.name.lower().endswith(('.xlsx', '.xls')):
        df_clients = pd.read_excel(uploaded)
    else:
        df_clients = pd.read_csv(uploaded)

    # Ajout du mapping interactif des colonnes
    cols = [c for c in df_clients.columns if isinstance(c, str)]
    st.write("**Colonnes détectées :**", cols)
    addr_cols = st.multiselect("Colonnes Adresse (sélection multiple)", options=cols, default=[c for c in cols if 'adresse' in c.lower() or 'voie' in c.lower() or 'rue' in c.lower()])
    cp_col = st.selectbox("Colonne Code Postal", options=[''] + cols, index=0)
    ville_col = st.selectbox("Colonne Ville", options=[''] + cols, index=0)
    comp_cols = [c for c in cols if 'compl' in c.lower()]
    comp_col = st.selectbox("Colonne Complément (facultatif)", options=[''] + comp_cols, index=0)

    # Construction de l'adresse complète
    df_clients['_full_address'] = ''
    for c in addr_cols:
        df_clients['_full_address'] += df_clients[c].fillna('').astype(str) + ' '
    if comp_col:
        df_clients['_full_address'] += df_clients[comp_col].fillna('').astype(str) + ' '
    if cp_col:
        df_clients['_full_address'] += df_clients[cp_col].fillna('').astype(str) + ' '
    if ville_col:
        df_clients['_full_address'] += df_clients[ville_col].fillna('').astype(str)

    # Géocodage des adresses
    lats, lons = [], []
    for addr in df_clients['_full_address']:
        lat, lon = geocode(addr)
        lats.append(lat)
        lons.append(lon)
        time.sleep(1)
    df_clients['Latitude'] = lats
    df_clients['Longitude'] = lons

    # Chargement des tournées
    df_tournees = load_tournees()

    # Attribution de la tournée la plus proche ou HZ
    assigned = []
    dist_min_list = []
    for _, cli in df_clients.iterrows():
        best = (None, float('inf'))
        cli_lat, cli_lon = cli['Latitude'], cli['Longitude']
        if pd.notna(cli_lat) and pd.notna(cli_lon):
            for _, tour in df_tournees.iterrows():
                d = distance_haversine(cli_lat, cli_lon, tour['Latitude'], tour['Longitude'])
                if d < best[1]:
                    best = (tour['Tournée'], d)
        # Si distance minimale > seuil, c'est hors zone
        if best[1] == float('inf') or best[1] > seuil_km:
            assigned.append('HZ')
            dist_min_list.append(best[1] if best[1] != float('inf') else None)
        else:
            assigned.append(best[0])
            dist_min_list.append(best[1])

    df_clients['Tournée attribuée'] = assigned
    df_clients['Distance (km)'] = dist_min_list

    # Affichage et téléchargement
    st.dataframe(df_clients)
    csv = df_clients.drop(columns=['_full_address']).to_csv(index=False)
    st.download_button("Télécharger le fichier enrichi", data=csv, file_name="clients_tournees_attribues.csv")

if __name__ == "__main__":
    main()
