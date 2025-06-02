import streamlit as st
import pandas as pd
import requests
import time
import io
from math import radians, sin, cos, sqrt, atan2
from unidecode import unidecode
import numpy as np

# --- Configuration ---
USER_AGENT = "TourneeLocator/1.0 (contact@votredomaine.com)"
TOURNEES_FILE = "Base_tournees_KML_coordonnees.xlsx"

# --- Fonctions utilitaires ---
def distance_haversine(lat1, lon1, lat2, lon2):
    """
    Calcule la distance en kilomètres entre deux points GPS.
    """
    R = 6371.0  # Rayon de la Terre en km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def clean_address(addr: str) -> str:
    """
    Nettoie l'adresse en :
      - supprimant accents/diacritiques (unidecode)
      - remplaçant abréviations courantes
      - supprimant doublons consécutifs
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
        # Attente pour respecter la règle 1 requête/sec de Nominatim
        time.sleep(1)
    return None, None

@st.cache_data
def load_tournees_with_nn_thresholds():
    """
    Charge la base des tournées (TOURNEES_FILE) contenant les colonnes :
      - 'Latitude'  (float)
      - 'Longitude' (float)
      - 'Tournée'   (string)
    Pour chaque tournée :
      1) extrait tous les points historiques (lat, lon)
      2) calcule pour chaque point la distance au plus proche voisin dans la même tournée
      3) construit un seuil = 90ème percentile de ces distances nearest‐neighbor
    Retourne :
      - df_ref           : DataFrame brute (with 'Latitude','Longitude','Tournée')
      - route_points_dict: { nom_tournée: np.array([[lat1,lon1], [lat2,lon2], ...]) }
      - thresholds_dict   : { nom_tournée: seuil_90_percentile_km }
    """
    df_ref = pd.read_excel(TOURNEES_FILE)
    route_points_dict = {}
    thresholds_dict = {}

    for name, grp in df_ref.groupby("Tournée"):
        # Tableau de shape (n_points, 2)
        pts = np.vstack([grp["Latitude"].values, grp["Longitude"].values]).T  # [[lat, lon], ...]
        route_points_dict[name] = pts

        # Si la tournée n'a qu'un seul point, on met un seuil très petit (par ex. 0.1 km)
        if pts.shape[0] == 1:
            thresholds_dict[name] = 0.1
            continue

        # Calculer la distance du plus proche voisin pour chaque point
        nn_distances = []
        for i in range(pts.shape[0]):
            lat_i, lon_i = pts[i]
            # Calcul des distances à tous les autres points de la même tournée
            dists = distance_haversine(
                lat_i, lon_i,
                pts[:, 0], pts[:, 1]
            )
            # On met distance=inf pour le même point (index i) pour ne pas se minorer
            dists[i] = np.inf
            nn_distances.append(dists.min())
        # Seuil = 90ème percentile des distances nearest‐neighbor
        seuil = np.percentile(nn_distances, 90)
        # S'assurer d'avoir un seuil minimal raisonnable (ex. 0.1 km)
        thresholds_dict[name] = max(seuil, 0.1)

    return df_ref, route_points_dict, thresholds_dict

def main():
    st.title("Attribution Automatique des Tournées PACA")
    st.write("1) Uploade ton fichier clients (Adresse, CP, Ville…)\n"
             "2) Laisse le système déterminer automatiquement la tournée la plus appropriée\n"
             "3) Télécharge le résultat (.xlsx)")

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

    # --- 4) Chargement des tournées historiques + seuils automatiques ---
    df_ref, route_points_dict, thresholds_dict = load_tournees_with_nn_thresholds()

    # --- 5) Attribution par plus proche voisin avec seuil dynamiques ---
    st.write("🚚 Attribution des tournées…")
    progress_attr = st.progress(0)
    attribs = []
    for i, row in enumerate(df.itertuples()):
        latc, lonc = getattr(row, "Latitude"), getattr(row, "Longitude")
        if pd.isna(latc) or pd.isna(lonc):
            attribs.append("")  # client non géocodé → on laisse vide
        else:
            # Calcul de la distance min du client à chaque point historique de chaque tournée
            best_route = ""
            best_distance = float("inf")

            for route_name, pts in route_points_dict.items():
                # pts.shape = (n_points_historic, 2)
                # Calculer distances h-to-all route-points d’un coup
                dists = distance_haversine(
                    latc, lonc,
                    pts[:, 0], pts[:, 1]
                )
                min_dist_to_route_points = dists.min()
                # Conserver la plus petite parmi toutes les tournées
                if min_dist_to_route_points < best_distance:
                    best_distance = min_dist_to_route_points
                    best_route = route_name

            # On compare best_distance au seuil de best_route
            seuil_route = thresholds_dict.get(best_route, 0.1)
            if best_distance <= seuil_route:
                attribs.append(best_route)
            else:
                attribs.append("")  # trop loin → vide

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

