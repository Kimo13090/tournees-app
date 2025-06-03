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

# CHEMIN VERS VOTRE KMZ (vérifiez que le nom de fichier est EXACT ici)
KMZ_TOURNEES_FILE = "abonnes_portes_analyste_tournee.kmz"

# Facteur nearest­-neighbor pour tolérance
NN_THRESHOLD_FACTOR = 2.0

# Buffer autour du convex hull (en degrés, ≃ 100 m)
HULL_BUFFER_DEGREES = 0.001

# ------------------------------------------------------------------------------
#                          FONCTIONS UTILITAIRES
# ------------------------------------------------------------------------------

def distance_haversine(lat1, lon1, lat2, lon2):
    """Calcule la distance en km entre deux points GPS."""
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def distance_haversine_array(lat0, lon0, lat_array, lon_array):
    """
    Version vectorisée du Haversine : 
    renvoie un tableau de distances [dist(i)] entre (lat0,lon0) et chaque (lat_array[i], lon_array[i]).
    """
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
    """
    Nettoie l'adresse : 
    - supprime les accents (unidecode) 
    - remplace les abréviations courantes (bd→boulevard, av→avenue, res→résidence)
    - supprime les doublons consécutifs
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
      1) on essaye l'adresse brute + " France"
      2) on essaye l'adresse nettoyée + " France"
    Gestion du code 429 (backoff exponentiel).
    Retourne (lat, lon) ou (None, None) en cas d'échec.
    """
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
    Lit un KMZ, extrait le KML et construit un dict de tournées :
    { 'NomTournée': np.array([[lat, lon], …]), … }.

    Gestion du cas où chaque tournée est dans un sous‐Folder “Autre” :
    1) Si un <Folder> a des <Placemark> DIRECTS → on l'utilise tel quel.
    2) Sinon, si ce <Folder> a exactement un sous‐<Folder> qui, LUI, contient
       des <Placemark> DIRECTS, on rattache ces derniers au nom du dossier parent.
    3) Sinon, on descend récursivement dans chaque sous‐Folder.
    """
    ns = {"kml": "http://www.opengis.net/kml/2.2"}

    with zipfile.ZipFile(kmz_path, "r") as kmz:
        kml_files = [fn for fn in kmz.namelist() if fn.lower().endswith(".kml")]
        if not kml_files:
            raise FileNotFoundError(f"Aucun fichier .kml trouvé dans {kmz_path}")
        kml_name = kml_files[0]
        with kmz.open(kml_name, "r") as f:
            tree = ET.parse(f)

    root = tree.getroot()
    route_points_dict = {}

    def extrait_placemarks(placemarks):
        """Retourne [(lat,lon), …] pour une liste d'<Placemark> passée."""
        pts = []
        for pm in placemarks:
            coord_txt = pm.find("kml:Point/kml:coordinates", ns)
            if coord_txt is None or not coord_txt.text:
                continue
            raw = coord_txt.text.strip()
            parts = raw.split(",")
            try:
                lon = float(parts[0]); lat = float(parts[1])
                pts.append((lat, lon))
            except ValueError:
                continue
        return pts

    def process_folder(folder_elem):
        """
        Fonction récursive. Pour un élément <Folder> donné, on fait :
        1) Si folder_elem a des <Placemark> DIRECTS → on crée une tournée là.
        2) Sinon, si folder_elem a exactement un sous‐Folder qui possède des <Placemark> DIRECTS,
           alors on rattache ces placemarks au nom du dossier parent.
        3) Sinon, on descend dans chacun de ses sous‐Folder (récursion).
        """
        placemarks_direct = folder_elem.findall("kml:Placemark", ns)
        if placemarks_direct:
            # 1) Cas de placemark direct
            name_elem = folder_elem.find("kml:name", ns)
            if name_elem is None or not name_elem.text:
                return
            tourn_name = name_elem.text.strip()
            coords = extrait_placemarks(placemarks_direct)
            if coords:
                route_points_dict[tourn_name] = np.array(coords, dtype=float)
            return

        # 2) Pas de placemark direct → regarder si 1 seul sous‐folder a des placemarks
        subfolders = folder_elem.findall("kml:Folder", ns)
        if len(subfolders) == 1:
            sous = subfolders[0]
            placemarks_sous = sous.findall("kml:Placemark", ns)
            if placemarks_sous:
                name_parent = folder_elem.find("kml:name", ns)
                if name_parent is None or not name_parent.text:
                    return
                tourn_name = name_parent.text.strip()
                coords = extrait_placemarks(placemarks_sous)
                if coords:
                    route_points_dict[tourn_name] = np.array(coords, dtype=float)
                return
            # Si ce sous-folder unique n'a pas de placemark direct, on descend dedans
            process_folder(sous)
            return

        # 3) Sinon, on descend normalement dans chacun de ses sous‐folders
        for subf in subfolders:
            process_folder(subf)

    # On lance la récursion sur TOUS les <Folder> dans le KML (quel que soit le niveau)
    for folder_root in root.findall(".//kml:Folder", ns):
        process_folder(folder_root)

    return route_points_dict

@st.cache_data
def load_tournees_with_nn_thresholds():
    """
    Charge les tournées via load_points_from_kmz, puis calcule pour chaque tournée :
    - convex hull + buffer (HULL_BUFFER_DEGREES)
    - seuil NN = 90ᵉ percentile des plus petites distances entre points
    Renvoie 3 dicts : route_points_dict, thresholds_dict, hulls_dict.
    """
    route_points_dict = load_points_from_kmz(KMZ_TOURNEES_FILE)

    thresholds_dict = {}
    hulls_dict = {}

    for name, pts in route_points_dict.items():
        # (a) nearest-neighbor 90ᵉ percentile
        if pts.shape[0] <= 1:
            thresholds_dict[name] = 0.1
        else:
            nn_distances = []
            for i in range(len(pts)):
                lat_i, lon_i = pts[i, 0], pts[i, 1]
                dists = distance_haversine_array(lat_i, lon_i, pts[:, 0], pts[:, 1])
                dists[i] = np.inf
                nn_distances.append(dists.min())
            seuil = np.percentile(nn_distances, 90)
            thresholds_dict[name] = max(seuil, 0.1)

        # (b) convex hull + buffer
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
        "1) Upload ton fichier clients (Adresse, CP, Ville…)\n"
        "2) L’app géocode, compare aux tournées du KMZ, et attribue la tournée la plus proche\n"
        "3) Télécharge le résultat (.xlsx)"
    )

    # 1) Upload du fichier client
    uploaded = st.file_uploader("Fichier Excel/CSV", type=["xlsx", "xls", "csv"])
    if not uploaded:
        return

    # 2) Détection de la ligne d’en-tête
    raw = pd.read_excel(uploaded, header=None)
    header_idx = 0
    for i, row in raw.iterrows():
        txt = " ".join(map(str, row.tolist())).lower()
        if any(k in txt for k in ("adresse", "cp", "codepostal", "ville")):
            header_idx = i
            break

    df = pd.read_excel(uploaded, header=header_idx)
    st.write("Colonnes détectées dans le fichier client :", list(df.columns))

    # 3) Choix auto/manu des colonnes Adresse, CP, Ville
    addr_cols = [
        c for c in df.columns
        if any(w in c.lower() for w in ("adresse", "voie", "rue", "route", "chemin"))
    ]
    cp_candidates = [c for c in df.columns if "codepostal" in c.lower() or c.lower() == "cp"]
    ville_candidates = [c for c in df.columns if "ville" in c.lower()]

    if not addr_cols:
        st.warning("📢 Impossible de détecter automatiquement la colonne ‘Adresse’. Choisissez-la manuellement.")
        choix_addr = st.selectbox("Quelle colonne contient l’adresse ?", options=list(df.columns))
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

    # 4) Construction du champ « _full_address »
    df["_full_address"] = ""
    for c in addr_cols + ([cp_col] if cp_col else []) + ([ville_col] if ville_col else []):
        if c:
            df["_full_address"] += df[c].fillna("").astype(str) + " "
    df["_full_address"] = df["_full_address"].apply(lambda x: unidecode(x.strip()))

    # 5) Géocodage avec barre de progression
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
        st.error("❌ Aucune adresse n’a été géocodée avec succès ! Vérifiez vos colonnes Adresse/CP/Ville ou la qualité des données.")
        return
    st.success(f"✅ Géocodage terminé ({n_valid}/{total} adresses valides)")

    # 6) Chargement du KMZ → extraction des tournées + calcul seuils & hulls
    try:
        route_points_dict, thresholds_dict, hulls_dict = load_tournees_with_nn_thresholds()
    except Exception as e:
        st.error(f"❌ Impossible de charger « {KMZ_TOURNEES_FILE} » ou d’en extraire les tournées :\n{e}")
        return

    n_tournees = len(route_points_dict)
    st.write(f"📂 {n_tournees} tournées extraites depuis le KMZ.")
    if n_tournees == 0:
        st.error("❌ Le KMZ ne contient aucune tournée valide. Vérifiez la structure (Folders avec Placemark).")
        return

    st.write("Exemple des tournées (jusqu’à 5) :")
    idx = 0
    for rn, pts in route_points_dict.items():
        st.write(f"   • {rn} → {pts.shape[0]} points")
        idx += 1
        if idx >= 5:
            break

    # 7) Attribution des tournées (buffer et nearest-neighbor)
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

        # 7.1) Si le point est DANS un convex hull bufferisé → on attribue cette tournée
        for route_name, hull_buf in hulls_dict.items():
            if hull_buf.contains(pt):
                choix = route_name
                break

        # 7.2) Sinon, fallback nearest-neighbor
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

    # 8) Export final en .xlsx
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
