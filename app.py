import streamlit as st
import pandas as pd
import requests
import time
import io
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from unidecode import unidecode
from shapely.geometry import MultiPoint, Point

# ------------------------------------------------------------------------------
#                            CONFIGURATION GLOBALE
# ------------------------------------------------------------------------------
USER_AGENT = "TourneeLocator/1.0 (contact@votredomaine.com)"
TOURNEES_FILE = "Base_tournees_KML_coordonnees.xlsx"

# Facteur multiplicateur sur le seuil nearest‐neighbor 90 %
NN_THRESHOLD_FACTOR = 1.5

# Tampon minimal pour le convex hull (en degrés, ≃ 50 m à l’équateur)
HULL_BUFFER_DEGREES = 0.0005  

# ------------------------------------------------------------------------------
#                          FONCTIONS UTILITAIRES
# ------------------------------------------------------------------------------
def distance_haversine(lat1, lon1, lat2, lon2):
    """
    Distance (en km) entre deux points GPS scalar (lat1, lon1) et (lat2, lon2).
    """
    R = 6371.0  # rayon moyen de la Terre en km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def distance_haversine_array(lat0, lon0, lat_array, lon_array):
    """
    Version vectorisée pour calculer la distance (en km) entre
    un point (lat0, lon0) et un array de points (lat_array, lon_array).
    Renvoie un numpy.ndarray de même longueur que lat_array.
    """
    R = 6371.0
    # Conversion en radians
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)
    lat_rad = np.radians(lat_array)
    lon_rad = np.radians(lon_array)

    dlat = lat_rad - lat0_rad
    dlon = lon_rad - lon0_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat0_rad) * np.cos(lat_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c  # tableau de distances

def clean_address(addr: str) -> str:
    """
    Nettoie l'adresse en :
      - Supprimant les accents/diacritiques (unidecode)
      - Remplaçant certaines abréviations (bd→boulevard, av→avenue, res→résidence)
      - Supprimant les doublons consécutifs
    """
    s = unidecode(addr or "")
    tokens = s.split()
    cleaned = []
    prev = None
    for t in tokens:
        tl = t.lower().strip(".,")
        if tl in ("bd", "bld", "boul"):
            t = "boulevard"
        elif tl in ("av", "av.", "aven"):
            t = "avenue"
        elif tl in ("res", "res."):
            t = "residence"
        if t.lower() != prev:
            cleaned.append(t)
            prev = t.lower()
    return " ".join(cleaned)

@st.cache_data(show_spinner=False)
def geocode(address: str):
    """
    Géocode une adresse via Nominatim :
      1) essai sur l'adresse brute
      2) essai sur l'adresse nettoyée (clean_address)
    Gère le code 429 avec backoff exponentiel.
    Retourne (lat, lon) ou (None, None) si aucun résultat.
    """
    headers = {"User-Agent": USER_AGENT}
    for variant in (address, clean_address(address)):
        if not variant.strip():
            continue
        backoff = 1.0
        while True:
            try:
                r = requests.get(
                    "https://nominatim.openstreetmap.org/search",
                    params={"q": variant + " France", "format": "json", "limit": 1},
                    headers=headers,
                    timeout=5
                )
            except requests.RequestException:
                time.sleep(1)
                break

            if r.status_code == 200:
                data = r.json()
                if data:
                    d = data[0]
                    return float(d["lat"]), float(d["lon"])
                time.sleep(1)
                break

            elif r.status_code == 429:
                time.sleep(backoff)
                backoff = min(backoff * 2, 5.0)
                continue

            else:
                time.sleep(1)
                break

    return None, None

@st.cache_data
def load_tournees_with_nn_thresholds():
    """
    Charge la base des tournées (TOURNEES_FILE) contenant :
      - 'Latitude'  (float)
      - 'Longitude' (float)
      - 'Tournée'   (string)
    Pour chaque tournée N dans ce fichier, on :
      1) collecte tous ses points historiques (lat, lon) dans pts_N
      2) calcule pour chaque point i de pts_N la distance au plus proche voisin
      3) fixe le seuil_N = 90ᵉ percentile des distances nearest‐neighbor
      4) construit un convex hull bufferisé (petit buffer pour inclure les bords)

    Renvoie :
      - df_ref            : DataFrame brute (Latitude, Longitude, Tournée)
      - route_points_dict : { nom_tournée: array([[lat,lon], ...]) }
      - thresholds_dict   : { nom_tournée: seuil_N_km }
      - hulls_dict        : { nom_tournée: shapely.Polygon.buffered }
    """
    df_ref = pd.read_excel(TOURNEES_FILE)
    route_points_dict = {}
    thresholds_dict = {}
    hulls_dict = {}

    for name, grp in df_ref.groupby("Tournée"):
        # Tableau de points historiques shape=(n_points,2)
        pts = np.vstack([grp["Latitude"].values, grp["Longitude"].values]).T
        route_points_dict[name] = pts

        if pts.shape[0] <= 1:
            thresholds_dict[name] = 0.1
        else:
            nn_distances = []
            for i in range(pts.shape[0]):
                lat_i, lon_i = pts[i, 0], pts[i, 1]
                dists = distance_haversine_array(lat_i, lon_i, pts[:, 0], pts[:, 1])
                dists[i] = np.inf
                nn_distances.append(dists.min())
            seuil = np.percentile(nn_distances, 90)
            thresholds_dict[name] = max(seuil, 0.1)

        shapely_pts = [Point(lon, lat) for lat, lon in zip(grp["Latitude"], grp["Longitude"])]
        hull = MultiPoint(shapely_pts).convex_hull
        hulls_dict[name] = hull.buffer(HULL_BUFFER_DEGREES)

    return df_ref, route_points_dict, thresholds_dict, hulls_dict

# ------------------------------------------------------------------------------
#                                FONCTION PRINCIPALE
# ------------------------------------------------------------------------------
def main():
    st.title("Attribution Automatique des Tournées PACA")
    st.write(
        "1) Uploade ton fichier clients (Adresse, CP, Ville…)\n"
        "2) Le système détermine automatiquement la tournée la plus appropriée\n"
        "3) Télécharge le résultat (.xlsx)"
    )

    uploaded = st.file_uploader("Fichier Excel/CSV", type=["xlsx", "xls", "csv"])
    if not uploaded:
        return

    # 1) Détection automatique de la ligne d'en-tête
    raw = pd.read_excel(uploaded, header=None)
    header_idx = 0
    for i, row in raw.iterrows():
        txt = " ".join(map(str, row.tolist())).lower()
        if any(k in txt for k in ("adresse", "cp", "codepostal", "ville")):
            header_idx = i
            break

    df = pd.read_excel(uploaded, header=header_idx)
    st.write("Colonnes détectées :", list(df.columns))

    # 2) Construction du champ d'adresse complète (normalisée)
    addr_cols = [
        c for c in df.columns
        if any(w in c.lower() for w in ("adresse", "voie", "rue", "route", "chemin"))
    ]
    cp_col = next((c for c in df.columns if "codepostal" in c.lower() or c.lower() == "cp"), None)
    ville_col = next((c for c in df.columns if "ville" in c.lower()), None)

    df["_full_address"] = ""
    for c in addr_cols + ([cp_col] if cp_col else []) + ([ville_col] if ville_col else []):
        df["_full_address"] += df[c].fillna("").astype(str) + " "
    df["_full_address"] = df["_full_address"].apply(lambda x: unidecode(x))

    # 3) Géocodage avec barre de progression
    total = len(df)
    st.write(f"🔍 Géocodage de {total} adresses…")
    progress_geo = st.progress(0)
    lats, lons = [], []
    for i, addr in enumerate(df["_full_address"]):
        lat, lon = geocode(addr)
        lats.append(lat)
        lons.append(lon)
        progress_geo.progress((i + 1) / total)
    df["Latitude"] = lats
    df["Longitude"] = lons
    st.success("✅ Géocodage terminé")

    # 4) Chargement des tournées historiques + calcul des seuils et hulls
    df_ref, route_points_dict, thresholds_dict, hulls_dict = load_tournees_with_nn_thresholds()

    # 5) Attribution prioritaire par CONVEX HULL tamponné, puis fallback NN
    st.write("🚚 Attribution des tournées…")
    progress_attr = st.progress(0)
    attribs = []

    for i, row in enumerate(df.itertuples()):
        latc, lonc = getattr(row, "Latitude"), getattr(row, "Longitude")

        if pd.isna(latc) or pd.isna(lonc):
            attribs.append("")
            progress_attr.progress((i + 1) / total)
            continue

        pt = Point(lonc, latc)
        choix = ""

        # 5.1) Si le client est dans le hull.buffer, on l’affecte
        for route_name, hull_buf in hulls_dict.items():
            if hull_buf.contains(pt):
                choix = route_name
                break

        # 5.2) Sinon fallback nearest-neighbor
        if choix == "":
            best_route = ""
            best_dist = float("inf")
            for route_name, pts in route_points_dict.items():
                dists_to_pts = distance_haversine_array(latc, lonc, pts[:, 0], pts[:, 1])
                dmin = float(dists_to_pts.min())
                if dmin < best_dist:
                    best_dist = dmin
                    best_route = route_name

            seuil = thresholds_dict.get(best_route, 0.1) * NN_THRESHOLD_FACTOR
            if best_dist <= seuil:
                choix = best_route
            else:
                choix = ""

        attribs.append(choix)
        progress_attr.progress((i + 1) / total)

    df["Tournée attribuée"] = attribs
    st.success("✅ Attribution terminée")

    # 6) Export en .xlsx
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    st.download_button(
        "Télécharger le fichier enrichi (.xlsx)",
        buffer.getvalue(),
        file_name="clients_tournees.xlsx",
        mime="application/vnd.openxmlformats-officedocument-spreadsheetml.sheet"
    )

if __name__ == "__main__":
    main()
