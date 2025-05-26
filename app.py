import streamlit as st
import pandas as pd
import requests
import time
import io
from math import radians, sin, cos, sqrt, atan2

# --- Configuration ---
USER_AGENT = "TourneeLocator/1.0 (contact@votredomaine.com)"
TOURNEES_FILE = "Base_tournees_KML_coordonnees.xlsx"

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
def load_tournees_radii():
    """
    Charge la base des tournées et calcule pour chaque tournée
    le centroïde GPS et un rayon robuste (90e percentile des distances).
    Retourne un dict : tourn_e -> (centroid_lat, centroid_lon, radius_km).
    """
    df = pd.read_excel(TOURNEES_FILE)
    # Grouper par nom de tournée
    radii = {}
    for name, grp in df.groupby('Tournée'):
        lats = grp['Latitude'].astype(float).tolist()
        lons = grp['Longitude'].astype(float).tolist()
        # centroïde
        cent_lat = sum(lats) / len(lats)
        cent_lon = sum(lons) / len(lons)
        # distances au centroïde
        dists = [distance_haversine(cent_lat, cent_lon, lat, lon) for lat, lon in zip(lats, lons)]
        # rayon = 90e percentile
        radius = pd.Series(dists).quantile(0.9)
        radii[name] = (cent_lat, cent_lon, radius)
    return radii

@st.cache_data
def geocode(address: str):  # cache pour ne pas surcharger le service
    """Géocode une adresse via Nominatim, retourne (lat, lon) ou (None,None)."""
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": address, "format": "json", "limit": 1}
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=5)
        if resp.status_code == 200 and resp.json():
            d = resp.json()[0]
            return float(d['lat']), float(d['lon'])
    except Exception:
        pass
    return None, None

# nettoyage avancé
def clean_address(addr: str) -> str:
    # supprimer doublons consécutifs
    tokens = addr.split()
    cleaned = []
    prev = None
    for t in tokens:
        if t.lower() != prev:
            cleaned.append(t)
        prev = t.lower()
    s = ' '.join(cleaned)
    # remplacer abréviations courantes
    repl = {' bd ': ' boulevard ', ' av ': ' avenue ', ' res ': ' résidence ',
            ' rte ': ' route ', ' ch ': ' chemin '}
    for k, v in repl.items():
        s = s.replace(k, v)
    return s

# --- Application Streamlit ---
def main():
    st.title("Attribution Automatique des Tournées PACA")
    st.write("Upload un fichier clients (Adresse, CP, Ville...). L'app géocode, assigne la tournée la plus proche ou marque HZ hors zone.")

    uploaded = st.file_uploader("Fichier Excel/CSV", type=["xlsx", "xls", "csv"])
    if not uploaded:
        return

    # détection automatique de l'en-tête
    # on lit sans header pour trouver ligne contenant Adresse/Code Postal/Ville
    tmp = pd.read_excel(uploaded, header=None, nrows=10)
    header_idx = 0
    for i, row in tmp.iterrows():
        keys = ''.join(map(str, row.values)).lower()
        if 'adresse' in keys or 'voie' in keys or 'rue' in keys or 'code' in keys and 'postal' in keys or 'ville' in keys:
            header_idx = i
            break
    # relire avec ce header
    if uploaded.name.lower().endswith(('.xls', '.xlsx')):
        df = pd.read_excel(uploaded, header=header_idx)
    else:
        df = pd.read_csv(uploaded, header=header_idx)

    # repérer les colonnes adresse, cp, ville
    cols = [c for c in df.columns if isinstance(c, str)]
    adresse_cols = [c for c in cols if any(k in c.lower() for k in ['adresse', 'voie', 'rue', 'chemin', 'route', 'av', 'bd', 'res'])]
    cp_cols = [c for c in cols if 'code postal' in c.lower() or c.lower() == 'cp']
    ville_cols = [c for c in cols if 'ville' in c.lower()]

    # UI de sélection en cas de plusieurs
    adresse_sel = st.selectbox('Colonne Adresse principale', adresse_cols, index=0 if adresse_cols else None)
    cp_sel = st.selectbox('Colonne Code Postal', cp_cols, index=0 if cp_cols else None)
    ville_sel = st.selectbox('Colonne Ville', ville_cols, index=0 if ville_cols else None)

    # créer _full_address
    df['_full_address'] = (
        df[adresse_sel].fillna('').astype(str) + ' ' +
        df[cp_sel].fillna('').astype(str) + ' ' +
        df[ville_sel].fillna('').astype(str)
    ).str.strip()

    # nettoyage et géocodage
    lats, lons = [], []
    for addr in df['_full_address']:
        lat, lon = geocode(addr)
        if lat is None:
            # essayer nettoyage
            addr2 = clean_address(addr)
            lat, lon = geocode(addr2)
        lats.append(lat)
        lons.append(lon)
        time.sleep(1)
    df['Latitude'] = lats
    df['Longitude'] = lons

    # charger tournées et radii
    radii = load_tournees_radii()

    # attribution tournée ou HZ
    tours, dists = [], []
    for _, row in df.iterrows():
        latc, lonc = row['Latitude'], row['Longitude']
        best = (None, float('inf'))
        for tourn, (clat, clon, rad) in radii.items():
            if pd.notna(latc) and pd.notna(lonc):
                d = distance_haversine(latc, lonc, clat, clon)
                # ne considérer que si dans le rayon robuste
                if d <= rad and d < best[1]:
                    best = (tourn, d)
        tours.append(best[0] or 'HZ')
        dists.append(best[1] if best[0] else None)
    df['Tournée attribuée'] = tours
    df['Distance (km)'] = dists

    # afficher et proposer téléchargement Excel
    st.dataframe(df)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Clients')
    buffer.seek(0)
    st.download_button(
        label='Télécharger le fichier Excel',
        data=buffer,
        file_name='clients_tournees.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

if __name__ == '__main__':
    main()
