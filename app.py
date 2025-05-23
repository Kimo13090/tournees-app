import streamlit as st
import pandas as pd
import requests
import time
from math import radians, sin, cos, sqrt, atan2

# --- Configuration ---
USER_AGENT = "TourneeLocator/1.0 (ton_email@domaine.com)"

# --- Fonctions utilitaires ---
def distance_haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

@st.cache_data
def load_tournees():
    """Charge la base de tournées géolocalisées depuis le fichier Excel"""
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
    st.write("Téléverse ton fichier client (Excel/CSV), l'app détectera le bon en-tête, géocodera et attribuera la tournée la plus proche.")

    uploaded = st.file_uploader("Fichier Excel/CSV", type=["xlsx", "xls", "csv"] )
    if not uploaded:
        return

    # Lecture brute pour détecter la ligne d'en-tête
    raw = pd.read_excel(uploaded, header=None)
    # Cherche la première ligne contenant 'adresse' (insensible à la casse)
    header_row = None
    for i, row in raw.iterrows():
        if row.astype(str).str.contains('adresse', case=False, na=False).any():
            header_row = i
            break
    if header_row is None:
        st.error("Impossible de trouver une ligne d'en-tête contenant 'Adresse'. Vérifie ton fichier.")
        return

    # Relecture avec la bonne ligne d'en-tête
    df_clients = pd.read_excel(uploaded, header=header_row)
    # Nettoyage : supprime les lignes de métadonnées avant l'en-tête
    df_clients = df_clients.dropna(axis=0, how='all').reset_index(drop=True)

    st.write("Colonnes détectées :", list(df_clients.columns))

    # Normalisation des noms de colonnes
    cols_map = {col.lower().strip().replace(" ", ""): col for col in df_clients.columns}
    adresse_key    = cols_map.get('adresse') or cols_map.get('adresseclient') or ''
    complement_key = cols_map.get('complementdadresse') or cols_map.get('complémentdadresse') or ''
    cp_key         = cols_map.get('codepostal') or cols_map.get('cp') or ''
    ville_key      = cols_map.get('ville') or cols_map.get('commune') or ''

    if not adresse_key:
        st.error("Aucune colonne 'Adresse' détectée. Vérifie le nom de la colonne dans ton fichier.")
        return

    # Construction de l'adresse complète
    parts = []
    for key in (adresse_key, complement_key, cp_key, ville_key):
        if key and key in df_clients.columns:
            parts.append(df_clients[key].fillna('').astype(str))
        else:
            parts.append(pd.Series([''] * len(df_clients)))
    df_clients['_full_address'] = parts[0] + ' ' + parts[1] + ' ' + parts[2] + ' ' + parts[3]

    # Géocodage (cache pour ne pas répéter)
    lats, lons = [], []
    for addr in df_clients['_full_address']:
        lat, lon = geocode(addr)
        lats.append(lat)
        lons.append(lon)
        time.sleep(1)
    df_clients['Latitude']  = lats
    df_clients['Longitude'] = lons

    # Attribution des tournées
    df_tournees = load_tournees()
    assigned = []
    for _, client in df_clients.iterrows():
        best = (None, float('inf'))
        zone = client.get('Ville') or client.get('Commune') or ''
        subset = df_tournees[df_tournees['Zone'].str.lower() == str(zone).lower()] if zone else df_tournees
        for _, tour in subset.iterrows():
            if pd.notna(client['Latitude']) and pd.notna(client['Longitude']):
                d = distance_haversine(client['Latitude'], client['Longitude'], tour['Latitude'], tour['Longitude'])
                if d < best[1]: best = (tour['Tournée'], d)
        assigned.append(best[0] or 'Non trouvé')
    df_clients['Tournée attribuée'] = assigned

    st.dataframe(df_clients)
    st.download_button("Télécharger le fichier enrichi", df_clients.to_csv(index=False), "clients_tournees.csv")

if __name__ == '__main__':
    main()
