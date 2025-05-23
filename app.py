import streamlit as st
import pandas as pd
import requests
import time
from math import radians, sin, cos, sqrt, atan2

# --- Configuration ---
USER_AGENT = "TourneeLocator/1.0 (contact@votredomaine.com)"

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
    """Charge la base de référence des tournées (à placer au même niveau que app.py)"""
    return pd.read_excel("Base_tournees_KML_coordonnees.xlsx")

@st.cache_data
def geocode(address):
    """Géocode une adresse via Nominatim (OSM)"""
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": address, "format": "json", "limit": 1}
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, params=params, headers=headers)
    if resp.status_code == 200 and resp.json():
        result = resp.json()[0]
        return float(result['lat']), float(result['lon'])
    return None, None

# --- Début de l'app ---

def main():
    st.title("Attribution automatique des tournées PACA 🌍")
    st.write("Téléversez votre fichier client, sélectionnez les colonnes adaptées et lancez.")

    uploaded = st.file_uploader("Fichier clients (Excel/CSV)", type=["xlsx", "xls", "csv"])
    if not uploaded:
        st.info("En attente du fichier client...")
        return

    # Lecture temporaire sans header pour détecter l'en-tête réel
    tmp = pd.read_excel(uploaded, header=None)
    header_row = None
    for i, row in tmp.iterrows():
        if any(str(cell).lower().strip().startswith('adresse') for cell in row):
            header_row = i
            break
    if header_row is None:
        st.error("Impossible de détecter la ligne d'en-tête. Assurez-vous que l'une des lignes contient 'Adresse'.")
        return
    # Relire avec le header détecté
    uploaded.seek(0)
    if uploaded.name.lower().endswith(('xlsx','xls')):
        df = pd.read_excel(uploaded, header=header_row)
    else:
        df = pd.read_csv(uploaded, header=header_row)

    st.write("**Colonnes détectées :**", list(df.columns))

    cols = list(df.columns)

    # Sélection des colonnes adresse (multiselect)
    default_addr = [c for c in cols if isinstance(c, str) and any(k in c.lower() for k in ['voie','rue','adresse','chemin','lotissement'])]
    addr_cols = st.multiselect("Colonnes d'adresse (voie, rue, etc.)", options=cols, default=default_addr)

    # Sélection du code postal
    default_cp = next((c for c in cols if isinstance(c, str) and 'code' in c.lower() and 'postal' in c.lower()), None)
    cp_col = st.selectbox("Colonne Code Postal", options=[None]+cols, index=cols.index(default_cp) if default_cp in cols else 0)

    # Sélection de la ville
    default_ville = next((c for c in cols if isinstance(c, str) and 'ville' in c.lower()), None)
    ville_col = st.selectbox("Colonne Ville", options=[None]+cols, index=cols.index(default_ville) if default_ville in cols else 0)

    # Concaténation de l'adresse complète
    parts = []
    for col in addr_cols:
        parts.append(df[col].fillna("").astype(str))
    if cp_col:
        parts.append(df[cp_col].fillna("").astype(str))
    if ville_col:
        parts.append(df[ville_col].fillna("").astype(str))
    df['_full_address'] = parts[0] if parts else pd.Series([""]*len(df))
    for part in parts[1:]:
        df['_full_address'] += ' ' + part

    st.write("**Exemple d'adresse complète :**", df['_full_address'].head())

    # Géocodage
    lats, lons = [], []
    for addr in df['_full_address']:
        lat, lon = geocode(addr)
        lats.append(lat)
        lons.append(lon)
        time.sleep(1)  # Fair-use Nominatim
    df['Latitude'] = lats
    df['Longitude'] = lons

    # Chargement des tournées
    df_tournees = load_tournees()

    # Attribuer la tournée la plus proche
    assigned = []
    for _, row in df.iterrows():
        best_t, best_d = None, float('inf')
        if pd.notna(row['Latitude']) and pd.notna(row['Longitude']):
            for _, t in df_tournees.iterrows():
                d = distance_haversine(row['Latitude'], row['Longitude'], t['Latitude'], t['Longitude'])
                if d < best_d:
                    best_t, best_d = t['Tournée'], d
        assigned.append(best_t or "Non trouvé")
    df['Tournée attribuée'] = assigned

    # Affichage et téléchargement
    st.dataframe(df)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Télécharger le fichier enrichi .csv", data=csv, file_name="clients_avec_tournee.csv", mime='text/csv')

if __name__ == '__main__':
    main()
