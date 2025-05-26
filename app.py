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
    R = 6371  # Rayon de la Terre en km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

@st.cache_data
def geocode(address):
    """Géocode une adresse via Nominatim et renvoie (lat, lon) ou (None, None)."""
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": address, "format": "json", "limit": 1}
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        data = resp.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception:
        pass
    return None, None

@st.cache_data
def load_tournees():
    """Charge la base de tournées et calcule centroides et seuils dynamiques."""
    df = pd.read_excel("Base_tournees_KML_coordonnees.xlsx")
    # Centroides
    centroids = df.groupby("Tournée").agg({"Latitude": "mean", "Longitude": "mean"}).rename(columns={"Latitude": "centroid_lat", "Longitude": "centroid_lon"})
    # Seuil max (distance max point-centro)
    thresholds = {}
    for tour, grp in df.groupby("Tournée"):
        lat0, lon0 = centroids.loc[tour, ["centroid_lat", "centroid_lon"]]
        dists = grp.apply(lambda row: distance_haversine(lat0, lon0, row["Latitude"], row["Longitude"]), axis=1)
        thresholds[tour] = dists.max()
    centroids["threshold_km"] = pd.Series(thresholds)
    centroids = centroids.reset_index()
    return df, centroids

# --- Interface ---
st.title("Attribution Automatique des Tournées PACA")
st.write(
    "**1.** Téléversez votre fichier client (Excel/CSV) avec les colonnes Adresse, CP, Ville.",
    "**2.** Laissez l’app détecter et géocoder les adresses.",
    "**3.** Les tournées sont attribuées selon le plus proche centröïde avec seuil dynamique; si aucune tournée n’a de point de référence à portée, le client est marqué 'HZ'."
)

# Upload fichier client
uploaded = st.file_uploader("Fichier client (Excel/CSV)", type=["xlsx", "xls", "csv"])
if not uploaded:
    st.stop()

# Lecture brut pour détection en-tête
df_raw = pd.read_excel(uploaded, header=None)
# trouver index de ligne contenant 'Adresse'
header_idx = 0
for idx, row in df_raw.iterrows():
    if row.astype(str).str.contains('Adresse', case=False, na=False).any():
        header_idx = idx
        break
# Relecture avec bon header
df = pd.read_excel(uploaded, header=header_idx)

# Affichage debug colonnes
st.write("Colonnes détectées :", list(df.columns))

# Sélection manuelle des champs
cols = [c for c in df.columns if isinstance(c, str)]
adresse_cols = st.multiselect("Colonnes à concaténer pour l'adresse :", cols, default=[c for c in cols if 'voie' in c.lower() or 'rue' in c.lower() or 'adresse' in c.lower()])
cp_col = st.selectbox("Colonne Code Postal :", cols, index=cols.index([c for c in cols if 'code' in c.lower() and 'postal' in c.lower()][0]) if any('code' in c.lower() and 'postal' in c.lower() for c in cols) else 0)
ville_col = st.selectbox("Colonne Ville :", cols, index=cols.index([c for c in cols if 'ville' in c.lower()][0]) if any('ville' in c.lower() for c in cols) else 0)

# Construction adresse complète
df['_full_address'] = ''
for col in adresse_cols:
    df['_full_address'] += df[col].fillna('').astype(str) + ' '
df['_full_address'] += df[cp_col].fillna('').astype(str) + ' '
df['_full_address'] += df[ville_col].fillna('').astype(str)

df['Latitude'], df['Longitude'] = zip(*df['_full_address'].apply(geocode))

# Chargement base tournées
df_ref, df_centroids = load_tournees()

# Attribution
assigned = []
distances = []
for _, client in df.iterrows():
    lat_c, lon_c = client['Latitude'], client['Longitude']
    best_tour = 'HZ'
    best_dist = None
    if pd.notna(lat_c) and pd.notna(lon_c):
        # calcul des distances aux centroides
        df_centroids['dist'] = df_centroids.apply(lambda r: distance_haversine(lat_c, lon_c, r['centroid_lat'], r['centroid_lon']), axis=1)
        df_c = df_centroids.sort_values('dist').reset_index(drop=True)
        # prendre la plus proche
        candidate = df_c.loc[0]
        best_dist = candidate['dist']
        # comparer au seuil dynamique
        if best_dist <= candidate['threshold_km']:
            best_tour = candidate['Tournée']
    distances.append(best_dist)
    assigned.append(best_tour)

df['Tournée attribuée'] = assigned
df['Distance (km)'] = distances

# Affichage et téléchargement
st.dataframe(df)
st.download_button("Télécharger résultat", df.to_csv(index=False), "clients_tournees_attribues.csv")

