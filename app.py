import streamlit as st
import pandas as pd
import requests
import time
from math import radians, sin, cos, sqrt, atan2

# --- Configuration ---
USER_AGENT = "TourneeLocator/1.0 (ton_email@domaine.com)"

# --- Fonctions utilitaires ---
def distance_haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Rayon de la Terre en km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

@st.cache_data
def load_tournees():
    # Charge la base des tournées (xlsx ou csv)
    try:
        df = pd.read_excel("Base_tournees_KML_coordonnees.xlsx")
    except FileNotFoundError:
        df = pd.read_csv("Base_tournees_KML_coordonnees.csv")
    return df

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

# --- Application Streamlit ---
def main():
    st.title("Attribution Automatique des Tournées")
    st.write("Upload un fichier Excel/CSV, l'app détecte vos colonnes et attribue la tournée la plus proche ou par zone existante.")

    # Téléversement
    uploaded = st.file_uploader("Fichier Excel/CSV", type=["xlsx", "xls", "csv"])
    if not uploaded:
        return

    # Lecture brute pour normalisation de header
    df_raw = pd.read_excel(uploaded, header=None)
    header_row = 0
    for idx, row in df_raw.iterrows():
        if row.str.contains('adresse', case=False, na=False).any():
            header_row = idx
            break
    df_clients = pd.read_excel(uploaded, header=header_row)

    st.write("Colonnes détectées :", list(df_clients.columns))

    # Chargement des tournées
    df_tournees = load_tournees()

    # Si la colonne 'Zone' existe, on fait un simple merge
    if 'Zone' in df_clients.columns:
        df_result = df_clients.merge(
            df_tournees[['Zone','Tournée']],
            on='Zone', how='left', suffixes=('','_attribuee')
        )
        df_result['Tournée attribuée'] = df_result.get('Tournée_attribuee', df_result.get('Tournée'))
    else:
        # Mapping dynamique des champs pour concat adresse
        cols_map = {col.lower().strip().replace(" ", ""): col for col in df_clients.columns}
        addr_col = cols_map.get('adresse') or cols_map.get('adressecomplète') or ''
        comp_col = cols_map.get('complementdadresse') or cols_map.get('complémentdadresse') or ''
        cp_col = cols_map.get('codepostal') or cols_map.get('cp') or ''
        ville_col = cols_map.get('ville') or ''

        # Création de l'adresse complète
        parts = []
        for key in (addr_col, comp_col, cp_col, ville_col):
            if key and key in df_clients.columns:
                parts.append(df_clients[key].fillna('').astype(str))
            else:
                parts.append(pd.Series(['']*len(df_clients)))
        df_clients['_full_address'] = parts[0] + ' ' + parts[1] + ' ' + parts[2] + ' ' + parts[3]

        # Géocodage
        lats, lons = [], []
        for addr in df_clients['_full_address']:
            lat, lon = geocode(addr)
            lats.append(lat); lons.append(lon)
            time.sleep(1)
        df_clients['Latitude'] = lats
        df_clients['Longitude'] = lons

        # Attribution par plus proche
        assigned = []
        for _, client in df_clients.iterrows():
            best = (None, float('inf'))
            if pd.notna(client['Latitude']) and pd.notna(client['Longitude']):
                #Limiter à la zone si ville/CP dispo
                zone_val = client.get('Ville') or client.get('Code Postal') or None
                subset = df_tournees
                if zone_val and 'Zone' in df_tournees.columns:
                    subset = df_tournees[df_tournees['Zone'] == zone_val]
                for _, tour in subset.iterrows():
                    d = distance_haversine(client['Latitude'], client['Longitude'], tour['Latitude'], tour['Longitude'])
                    if d < best[1]: best = (tour['Tournée'], d)
            assigned.append(best[0] or 'Non trouvé')
        df_clients['Tournée attribuée'] = assigned
        df_result = df_clients

    st.dataframe(df_result)
    csv = df_result.to_csv(index=False).encode('utf-8')
    st.download_button("Télécharger le fichier enrichi", csv, "clients_tournees.csv", "text/csv")

if __name__ == '__main__':
    main()
