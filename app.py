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
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def clean_address(addr: str) -> str:
    """Supprime les tokens consécutifs identiques et nettoie l'adresse."""
    tokens = addr.split()
    cleaned = []
    prev = None
    for t in tokens:
        if t.lower() != prev:
            cleaned.append(t)
        prev = t.lower()
    return " ".join(cleaned)


def expand_abbreviations(addr: str) -> str:
    """Remplace les abréviations courantes par leur forme complète."""
    mapping = {
        "\bbd\b": "Boulevard",
        "\bav\b": "Avenue",
        "\bres\b": "Résidence",
        "\bche\b": "Chemin",
        "\brte\b": "Route"
    }
    import re
    for abbr, full in mapping.items():
        addr = re.sub(abbr, full, addr, flags=re.IGNORECASE)
    return addr


def replace_synonyms(addr: str) -> str:
    """Tente des synonymes pour corriger des erreurs de type 'route' vs 'rue'."""
    synonyms = {
        " Route ": " Rue ",
        " Rte ": " Rue ",
        " Chemin ": " Rue "
    }
    for wrong, right in synonyms.items():
        addr = addr.replace(wrong, right)
    return addr


def geocode_address(addr: str):
    """Essaye de géocoder l'adresse, avec nettoyages et corrections successives."""
    variations = [addr,
                  clean_address(addr),
                  expand_abbreviations(clean_address(addr)),
                  replace_synonyms(expand_abbreviations(clean_address(addr)))]
    for var in variations:
        try:
            url = "https://nominatim.openstreetmap.org/search"
            params = {"q": var, "format": "json", "limit": 1}
            headers = {"User-Agent": USER_AGENT}
            resp = requests.get(url, params=params, headers=headers)
            if resp.status_code == 200 and resp.json():
                data = resp.json()[0]
                return float(data["lat"]), float(data["lon"]), var
        except Exception:
            pass
        time.sleep(1)
    return None, None, None


@st.cache_data
def load_tournees():
    df = pd.read_excel("Base_tournees_KML_coordonnees.xlsx")
    # calculer centroids et rayons
    centroids = []
    radii = []
    for name, group in df.groupby("Tournée"):
        lats = group["Latitude"]
        lons = group["Longitude"]
        centroid_lat = lats.mean()
        centroid_lon = lons.mean()
        # rayon max
        max_dist = max(distance_haversine(centroid_lat, centroid_lon, lat, lon)
                       for lat, lon in zip(lats, lons))
        centroids.append((name, centroid_lat, centroid_lon, max_dist))
    return centroids


def main():
    st.title("Attribution Automatique des Tournées PACA")
    st.write("Upload fichier clients (Adresse, CP, Ville...). L'app attribue ou marque HZ hors zone.")
    uploaded = st.file_uploader("Fichier Excel/CSV", type=["xlsx", "xls", "csv"])
    if not uploaded:
        return
    # lecture
    if uploaded.name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded, header=0)
    else:
        df = pd.read_csv(uploaded, header=0)

    st.write("Colonnes détectées :", list(df.columns))
    # sélection des colonnes
    cols = [c for c in df.columns if isinstance(c, str)]
    addr_cols = st.multiselect("Colonnes Adresse (voie, rue...) :", cols, default=[c for c in cols if any(k in c.lower() for k in ["voie","rue","chemin","av","bd"])][:2])
    cp_col = st.selectbox("Colonne Code Postal :", cols, index=cols.index(next((c for c in cols if "code" in c.lower()), cols[0])))
    ville_col = st.selectbox("Colonne Ville :", cols, index=cols.index(next((c for c in cols if "ville" in c.lower()), cols[0])))

    # concat adresse
    df['_full_address'] = df[addr_cols].astype(str).apply(lambda row: ' '.join(row.values), axis=1) + \
                         ' ' + df[cp_col].astype(str) + ' ' + df[ville_col].astype(str)

    centroids = load_tournees()
    # geocode et attribution
    tournee_assigne = []
    dist_list = []
    for addr in df['_full_address']:
        lat, lon, used = geocode_address(addr)
        if lat is None:
            tournee_assigne.append("HZ")
            dist_list.append(None)
            continue
        # calcul distances aux centroids
        best = (None, float('inf'))
        for name, clat, clon, radius in centroids:
            d = distance_haversine(lat, lon, clat, clon)
            if d <= radius and d < best[1]:
                best = (name, d)
        if best[0]:
            tournee_assigne.append(best[0])
            dist_list.append(round(best[1],2))
        else:
            tournee_assigne.append("HZ")
            dist_list.append(None)

    df['Tournée attribuée'] = tournee_assigne
    df['Distance km'] = dist_list

    # export Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    data = output.getvalue()
    st.download_button("Télécharger Excel avec Tournées", data, file_name="clients_tournees_attribues.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__ == "__main__":
    main()
