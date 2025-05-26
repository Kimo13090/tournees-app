import streamlit as st
import pandas as pd
import requests
import time
import io
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
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c


def clean_address(addr: str) -> str:
    """Nettoie et normalise une adresse pour améliorer le géocodage."""
    if not isinstance(addr, str):
        return ""
    # suppression des doublons consécutifs
    tokens = addr.split()
    cleaned = []
    prev = None
    for t in tokens:
        if t != prev:
            cleaned.append(t)
        prev = t
    s = " ".join(cleaned)
    # abréviations courantes
    repl = {
        r"\bbd\b": "boulevard",
        r"\bav\b": "avenue",
        r"\bres\b": "résidence",
        r"\brte\b": "rue",
        r"\bchemin\b": "chemin"
    }
    for k, v in repl.items():
        s = pd.Series([s]).str.replace(k, v, regex=True, case=False)[0]
    return s

@st.cache_data
def geocode(addr: str):
    """Retourne (lat, lon) via Nominatim ou (None, None) si échec."""
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": addr, "format": "json", "limit": 1}
    headers = {"User-Agent": USER_AGENT}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        if r.ok and r.json():
            d = r.json()[0]
            return float(d["lat"]), float(d["lon"])
    except Exception:
        pass
    return None, None

@st.cache_data
def load_tournees():
    df = pd.read_excel("Base_tournees_KML_coordonnees.xlsx")
    # calcul de centroides et rayons pour chaque tournée
    out = []
    for name, grp in df.groupby('Tournée'):
        lat_mean = grp['Latitude'].mean()
        lon_mean = grp['Longitude'].mean()
        # rayon max
        grp['dist'] = grp.apply(lambda r: distance_haversine(lat_mean, lon_mean, r['Latitude'], r['Longitude']), axis=1)
        radius = grp['dist'].max()
        out.append({
            'Tournée': name,
            'cent_lat': lat_mean,
            'cent_lon': lon_mean,
            'rayon_km': radius
        })
    return pd.DataFrame(out)

# --- Interface ---
st.set_page_config(page_title="Attribution Automatique des Tournées PACA", layout='wide')
st.title("🗺️ Attribution Automatique des Tournées PACA")
st.write("Upload un fichier clients (Adresse, CP, Ville, etc.). L'app nettoie, géocode, et attribue la tournée ou marque HZ.")

uploaded = st.file_uploader("Fichier Excel/CSV à traiter", type=["xlsx","xls","csv"])
if not uploaded:
    st.stop()

# Lecture avec détection de l'en-tête
raw = pd.read_excel(uploaded, header=None)
header_idx = 0
for idx, row in raw.iterrows():
    if row.astype(str).str.contains('adresse', case=False, na=False).any():
        header_idx = idx
        break
# relire avec le bon header
if str(uploaded.name).lower().endswith(('xls','xlsx')):
    df = pd.read_excel(uploaded, header=header_idx)
else:
    df = pd.read_csv(uploaded, header=header_idx)

st.write("**Colonnes détectées (après header)** : ", list(df.columns))

# Mapping dynamique des colonnes
cols = [c for c in df.columns if isinstance(c, str)]
addr_cols = [c for c in cols if 'adresse' in c.lower() or 'voie' in c.lower() or 'rue' in c.lower() or 'chemin' in c.lower()]
cp_cols   = [c for c in cols if 'code' in c.lower() and 'postal' in c.lower() or c.lower()=='cp']
ville_cols = [c for c in cols if 'ville' in c.lower() or 'commune' in c.lower()]

# Sélection utilisateur si multiples
addr_sel = st.multiselect("Colonnes Adresse (voie, rue, chemin)", addr_cols, default=addr_cols)
cp_sel   = st.selectbox("Colonne Code Postal", cp_cols)
ville_sel = st.selectbox("Colonne Ville", ville_cols)

# Assemblage
df['_full_address'] = (
    df[addr_sel].fillna('').agg(' '.join, axis=1) + ' ' +
    df[cp_sel].astype(str).fillna('') + ' ' +
    df[ville_sel].fillna('')
)
# Nettoyage
df['_full_address'] = df['_full_address'].apply(clean_address)

# Géocodage
lats, lons = [], []
for addr in df['_full_address']:
    lat, lon = geocode(addr)
    if lat is None:
        # retry with raw addr
        lat, lon = geocode(addr)
    lats.append(lat); lons.append(lon)
    time.sleep(1)
df['Latitude'] = lats; df['Longitude'] = lons

# Chargement tournées
df_tour = load_tournees()

# Attribution
assigned, dists = [], []
for _, row in df.iterrows():
    best = ('HZ', None)
    if pd.notna(row['Latitude']):
        for _, t in df_tour.iterrows():
            d = distance_haversine(row['Latitude'], row['Longitude'], t['cent_lat'], t['cent_lon'])
            if d <= t['rayon_km']:
                best = (t['Tournée'], d)
                break
    assigned.append(best[0]); dists.append(best[1])

df['Tournée attribuée'] = assigned
df['Distance (km)'] = dists

# Affichage et téléchargement
st.dataframe(df)

# Excel download
buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
    df.to_excel(writer, index=False, sheet_name='Résultat')
    writer.save()
buffer.seek(0)

st.download_button("Télécharger résultat (.xlsx)", data=buffer, 
                   file_name="clients_tournees_attribués.xlsx", 
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
