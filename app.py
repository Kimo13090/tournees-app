import streamlit as st
import pandas as pd
import requests
import time
import io
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from unidecode import unidecode
from shapely.geometry import MultiPoint, Point
import zipfile
import xml.etree.ElementTree as ET

# ------------------------------------------------------------------------------
#                            CONFIGURATION GLOBALE
# ------------------------------------------------------------------------------
USER_AGENT = "TourneeLocator/1.0 (contact@votredomaine.com)"

# Assurez-vous que ce chemin est EXACTEMENT celui du KMZ dans votre repo
KMZ_TOURNEES_FILE = "abonnes_portes_analyste_tournee.kmz"

# Facteur nearest‐neighbor (tolérance supplémentaire)
NN_THRESHOLD_FACTOR = 2.0

# Tampon pour le convex hull (en degrés, ≃ 100 m)
HULL_BUFFER_DEGREES = 0.001

# ------------------------------------------------------------------------------
#                          FONCTIONS UTILITAIRES
# ------------------------------------------------------------------------------

def distance_haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def distance_haversine_array(lat0, lon0, lat_array, lon_array):
    R = 6371.0
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)
    lat_rad = np.radians(lat_array)
    lon_rad = np.radians(lon_array)

    dlat = lat_rad - lat0_rad
    dlon = lon_rad - lon0_rad
    a = np.sin(dlat/2)**2 + np.cos(lat0_rad) * np.cos(lat_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def clean_address(addr: str) -> str:
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
    headers = {"User-Agent": USER_AGENT}
    variants = [address, clean_address(address)]
    for var in variants:
        if not var.strip():
            continue
        backoff = 1.0
        while True:
            try:
                resp = requests.get(
                    "https://nominatim.openstreetmap.org/search",
                    params={"q": var + " France", "format": "json", "limit": 1},
                    headers=headers,
                    timeout=5
                )
            except requests.RequestException:
                time.sleep(1)
                break

            if resp.status_code == 200:
                data = resp.json()
                if data:
                    d0 = data[0]
                    return float(d0["lat"]), float(d0["lon"])
                time.sleep(1)
                break

            elif resp.status_code == 429:
                time.sleep(backoff)
                backoff = min(backoff * 2, 5.0)
                continue
            else:
                time.sleep(1)
                break

    return None, None

def load_points_from_kmz(kmz_path: str):
    """
    Lit le KMZ, extrait le (premier) KML, et fouille récursivement chaque <Folder>
    pour trouver les tournées (un Folder est considéré comme "tournée" s'il contient
    au moins un <Placemark> direct). Si un Folder n'a pas de Placemark direct,
    on descend d'un cran via ses sous‐folders.
    Renvoie un dict : { 'NomTournée': np.array([[lat, lon], ...]), ... }.
    """
    ns = {"kml": "http://www.opengis.net/kml/2.2"}

    # 1) Extraire le KML depuis le KMZ
    with zipfile.ZipFile(kmz_path, "r") as kmz:
        kml_files = [fn for fn in kmz.namelist() if fn.lower().endswith(".kml")]
        if not kml_files:
            raise FileNotFoundError(f"Aucun fichier .kml trouvé dans {kmz_path}")
        kml_name = kml_files[0]
        with kmz.open(kml_name, "r") as f:
            tree = ET.parse(f)

    root = tree.getroot()
    route_points_dict = {}

    def process_folder(folder_elem):
        # 1) Chercher tout <Placemark> ENFANT DIRECT du folder
        placemarks_direct = folder_elem.findall("kml:Placemark", ns)
        if placemarks_direct:
            name_elem = folder_elem.find("kml:name", ns)
            if name_elem is None or not name_elem.text:
                return
            tourn_name = name_elem.text.strip()

            pts_list = []
            for pm in placemarks_direct:
                coord_elem = pm.find("kml:Point/kml:coordinates", ns)
                if coord_elem is None or not coord_elem.text:
                    continue
                raw = coord_elem.text.strip()
                parts = raw.split(",")
                try:
                    lon = float(parts[0])
                    lat = float(parts[1])
                    pts_list.append((lat, lon))
                except ValueError:
                    continue

            if pts_list:
                route_points_dict[tourn_name] = np.array(pts_list, dtype=float)
            return  # on ne descend pas plus bas, on considère ce folder comme “terminal”

        # 2) Sinon, ce Folder n'a pas de Placemark direct → explorer ses sous‐folders
        for subf in folder_elem.findall("kml:Folder", ns):
            process_folder(subf)

    # 3) On lance la récursion sur TOUTES les occurrences de <Folder>
    for folder_root in root.findall(".//kml:Folder", ns):
        process_folder(folder_root)

    return route_points_dict

@st.cache_data
def load_tournees_with_nn_thresholds():
    """
    Charge tous les points depuis le KMZ (via load_points_from_kmz), puis calcule :
      - convex hull + buffer (HULL_BUFFER_DEGREES)
      - 90ᵉ percentile nearest-neighbor (NN threshold)
    Renvoie 3 dict : route_points_dict, thresholds_dict, hulls_dict.
    """
    route_points_dict = load_points_from_kmz(KMZ_TOURNEES_FILE)

    thresholds_dict = {}
    hulls_dict = {}

    for name, pts in route_points_dict.items():
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

        shapely_pts = [Point(lon, lat) for lat, lon in pts]
        hull = MultiPoint(shapely_pts).convex_hull
        hulls_dict[name] = hull.buffer(HULL_BUFFER_DEGREES)

    return route_points_dict, thresholds_dict, hulls_dict

# ------------------------------------------------------------------------------
#                                FONCTION PRINCIPALE
# ------------------------------------------------------------------------------
def main():
    st.title("Attribution Automatique des Tournées PACA (KMZ)")

    st.write(
        "1) Uploade ton fichier clients (Adresse, CP, Ville…)\n"
        "2) L'app géocode, compare aux tournées du KMZ, et attribue la tournée la plus proche\n"
        "3) Télécharge le résultat (.xlsx)"
    )

    # --------------------------------------------------------------------------
    # 1) Upload du fichier clients
    # --------------------------------------------------------------------------
    uploaded = st.file_uploader("Fichier Excel/CSV", type=["xlsx", "xls", "csv"])
    if not uploaded:
        return

    # --------------------------------------------------------------------------
    # 2) Détection de la ligne d'en-tête (auto. + fallback)
    # --------------------------------------------------------------------------
    raw = pd.read_excel(uploaded, header=None)
    header_idx = 0
    for i, row in raw.iterrows():
        txt = " ".join(map(str, row.tolist())).lower()
        if any(k in txt for k in ("adresse", "cp", "codepostal", "ville")):
            header_idx = i
            break

    df = pd.read_excel(uploaded, header=header_idx)
    st.write("Colonnes détectées dans le fichier client :", list(df.columns))

    # --------------------------------------------------------------------------
    # 3) Détection auto. / sélection manuelle des colonnes Adresse, CP, Ville
    # --------------------------------------------------------------------------
    addr_cols = [
        c for c in df.columns
        if any(w in c.lower() for w in ("adresse", "voie", "rue", "route", "chemin"))
    ]
    cp_candidates = [c for c in df.columns if "codepostal" in c.lower() or c.lower() == "cp"]
    ville_candidates = [c for c in df.columns if "ville" in c.lower()]

    if not addr_cols:
        st.warning("📢 Impossible de détecter automatiquement la colonne ‘Adresse’. Choisissez-la manuellement.")
        choix_addr = st.selectbox("Quelle colonne contient l'adresse ?", options=list(df.columns))
        addr_cols = [choix_addr]

    if not cp_candidates:
        st.warning("📢 Impossible de détecter la colonne ‘Code Postal’. Choisissez-la manuellement (ou laissez vide).")
        cp_candidates = [""] + list(df.columns)
        choix_cp = st.selectbox("Quelle colonne contient le code postal ? (laisser vide si non présent)", options=cp_candidates)
        cp_col = choix_cp if choix_cp != "" else None
    else:
        cp_col = cp_candidates[0]

    if not ville_candidates:
        st.warning("📢 Impossible de détecter la colonne ‘Ville’. Choisissez-la manuellement (ou laissez vide).")
        ville_candidates = [""] + list(df.columns)
        choix_ville = st.selectbox("Quelle colonne contient la ville ? (laisser vide si non présent)", options=ville_candidates)
        ville_col = choix_ville if choix_ville != "" else None
    else:
        ville_col = ville_candidates[0]

    st.write(f"→ Colonnes sélectionnées : Adresse={addr_cols}, CP={cp_col}, Ville={ville_col}")

    # --------------------------------------------------------------------------
    # 4) Construction du champ « _full_address »
    # --------------------------------------------------------------------------
    df["_full_address"] = ""
    for c in addr_cols + ([cp_col] if cp_col else []) + ([ville_col] if ville_col else []):
        if c:
            df["_full_address"] += df[c].fillna("").astype(str) + " "
    df["_full_address"] = df["_full_address"].apply(lambda x: unidecode(x.strip()))

    # --------------------------------------------------------------------------
    # 5) Géocodage (affichage du progrès)
    # --------------------------------------------------------------------------
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

    n_valid = df[["Latitude", "Longitude"]].dropna().shape[0]
    if n_valid == 0:
        st.error("❌ Aucune adresse n’a pu être géocodée ! Vérifiez vos colonnes Adresse/CP/Ville.")
        return
    st.success(f"✅ Géocodage terminé ({n_valid}/{total} adresses valides)")

    # --------------------------------------------------------------------------
    # 6) Chargement du KMZ → extraction des points + calcul des seuils & hulls
    # --------------------------------------------------------------------------
    try:
        route_points_dict, thresholds_dict, hulls_dict = load_tournees_with_nn_thresholds()
    except Exception as e:
        st.error(f"❌ Impossible de charger « {KMZ_TOURNEES_FILE} » ou d’en extraire les tournées :\n{e}")
        return

    n_tournees = len(route_points_dict)
    st.write(f"📂 {n_tournees} tournées extraites depuis le KMZ.")
    if n_tournees == 0:
        st.error("❌ Le KMZ ne contient aucune tournée valide. Vérifiez la structure de votre fichier KMZ (Folders avec Placemark).")
        return

    st.write("Exemple des tournées (jusqu’à 5) :")
    idx = 0
    for rn, pts in route_points_dict.items():
        st.write(f"   • {rn} → {pts.shape[0]} points")
        idx += 1
        if idx >= 5:
            break

    # --------------------------------------------------------------------------
    # 7) Attribution des tournées (hull tamponné + fallback nearest‐neighbor)
    # --------------------------------------------------------------------------
    st.write("🚚 Attribution des tournées en cours…")
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

        # 7.1) Vérifier si le point est DANS un convex hull bufferisé
        for route_name, hull_buf in hulls_dict.items():
            if hull_buf.contains(pt):
                choix = route_name
                break

        # 7.2) Sinon, fallback nearest‐neighbor
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
    st.success("✅ Attribution des tournées terminée")

    # --------------------------------------------------------------------------
    # 8) Export en .xlsx
    # --------------------------------------------------------------------------
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    st.download_button(
        "📥 Télécharger le fichier enrichi (.xlsx)",
        buffer.getvalue(),
        file_name="clients_tournees.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if __name__ == "__main__":
    main()
