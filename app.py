import streamlit as st
import pandas as pd
import requests
import time
import io
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from unidecode import unidecode

# --- Configuration ---
USER_AGENT = "TourneeLocator/1.0 (contact@votredomaine.com)"
TOURNEES_FILE = "Base_tournees_KML_coordonnees.xlsx"

# --- Fonctions utilitaires ---

def distance_haversine(lat1, lon1, lat2, lon2):
    """
    Calcule la distance (en km) entre deux points GPS scalaires (lat1, lon1) et (lat2, lon2).
    """
    R = 6371.0  # Rayon de la Terre en km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def distance_haversine_array(lat0, lon0, lat_array, lon_array):
    """
    Version vectorisée de distance_haversine : calcule la distance (en km) entre
    un point (lat0, lon0) et un tableau de points (lat_array, lon_array) de même taille.
    """
    R = 6371.0
    # Convertir en radians
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)
    lat_rad = np.radians(lat_array)
    lon_rad = np.radians(lon_array)

    dlat = lat_rad - lat0_rad
    dlon = lon_rad - lon0_rad
    a = np.sin(dlat / 2)**2 + np.cos(lat0_rad) * np.cos(lat_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c  # renvoie un tableau de distances

def clean_address(addr: str) -> str:
    """
    Nettoie l'adresse :
      - Supprime les diacritiques (unidecode)
      - Développe quelques abréviations courantes (bd → boulevard, av → avenue, res → residence)
      - Supprime les doublons consécutifs de mots
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
      2) essai sur l'adresse nettoyée
    Retourne (lat, lon) ou (None, None) si échec.
    """
    headers = {"User-Agent": USER_AGENT}
    for variant in (address, clean_address(address)):
        if not variant.strip():
            continue
        try:
            r = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": variant, "format": "json", "limit": 1},
                headers=headers,
                timeout=5
            )
        except:
            continue
        if r.status_code == 200 and r.json():
            d = r.json()[0]
            return float(d["lat"]), float(d["lon"])
        time.sleep(1)  # Respect de la règle de 1requête/sec API Nominatim
    return None, None

@st.cache_data
def load_tournees_with_nn_thresholds():
    """
    Charge la base des tournées (TOURNEES_FILE), qui contient :
      - 'Latitude' (float)
      - 'Longitude'(float)
      - 'Tournée'   (string)
    Pour chaque tournée, on :
      1) extrait tous les points historiques (lat, lon)
      2) calcule pour chaque point l’écart le plus petit (nearest‐neighbor) au sein de la même tournée
      3) déduit le seuil de distance = 90ᵉ percentile de ces distances
    Retourne :
      - df_ref            : DataFrame brute avec les colonnes (Latitude, Longitude, Tournée)
      - route_points_dict : { nom_tournée: np.array([[lat, lon], ...]) }
      - thresholds_dict   : { nom_tournée: seuil_90_percentile_en_km }
    """
    df_ref = pd.read_excel(TOURNEES_FILE)
    route_points_dict = {}
    thresholds_dict = {}

    for name, grp in df_ref.groupby("Tournée"):
        pts = np.vstack([grp["Latitude"].values, grp["Longitude"].values]).T  # shape=(n_points, 2)
        route_points_dict[name] = pts

        # Si un seul point historique dans la tournée, seuil minimal 0.1 km
        if pts.shape[0] <= 1:
            thresholds_dict[name] = 0.1
            continue

        # Calcul des distances nearest‐neighbor pour chaque point
        nn_distances = []
        for i in range(pts.shape[0]):
            lat_i, lon_i = pts[i, 0], pts[i, 1]
            # distance du point i à tous les autres points de la même tournée
            dists = distance_haversine_array(lat_i, lon_i, pts[:, 0], pts[:, 1])
            # Pour ne pas prendre distance à soi-même
            dists[i] = np.inf
            nn_distances.append(dists.min())
        # Seuil = 90ᵉ percentile des distances nearest‐neighbor
        seuil = np.percentile(nn_distances, 90)
        thresholds_dict[name] = max(seuil, 0.1)

    return df_ref, route_points_dict, thresholds_dict

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

    # --- 1) Détection automatique de la ligne d'en-tête ---
    raw = pd.read_excel(uploaded, header=None)
    header_idx = 0
    for i, row in raw.iterrows():
        txt = " ".join(map(str, row.tolist())).lower()
        if any(k in txt for k in ("adresse", "cp", "codepostal", "ville")):
            header_idx = i
            break

    df = pd.read_excel(uploaded, header=header_idx)
    st.write("Colonnes détectées :", list(df.columns))

    # --- 2) Construction d'une adresse complète normalisée ---
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

    # --- 3) Géocodage avec barre de progression ---
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

    # --- 4) Chargement des tournées historiques + calcul des seuils dynamiques ---
    df_ref, route_points_dict, thresholds_dict = load_tournees_with_nn_thresholds()

    # --- 5) Attribution par distance minimale avec seuils dynamiques ---
    st.write("🚚 Attribution des tournées…")
    progress_attr = st.progress(0)
    attribs = []

    for i, row in enumerate(df.itertuples()):
        latc, lonc = getattr(row, "Latitude"), getattr(row, "Longitude")
        if pd.isna(latc) or pd.isna(lonc):
            attribs.append("")  # adresse non géocodée → on laisse vide
        else:
            best_route = ""
            best_dist = float("inf")

            # Pour chaque tournée historique, calculer la distance min du client
            for route_name, pts in route_points_dict.items():
                # pts[:,0] = array des latitudes, pts[:,1] = array des longitudes
                dists_to_points = distance_haversine_array(latc, lonc, pts[:, 0], pts[:, 1])
                min_dist_to_route = dists_to_points.min()

                if min_dist_to_route < best_dist:
                    best_dist = min_dist_to_route
                    best_route = route_name

            # On compare best_dist avec le seuil de la tournée retenue
            seuil_route = thresholds_dict.get(best_route, 0.1)
            if best_dist <= seuil_route:
                attribs.append(best_route)
            else:
                attribs.append("")  # trop éloigné → on laisse vide

        progress_attr.progress((i + 1) / total)

    df["Tournée attribuée"] = attribs
    st.success("✅ Attribution terminée")

    # --- 6) Export en .xlsx ---
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

