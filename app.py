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

# Chemin vers le KMZ contenant toutes vos tournées (abonnés/points historiques)
KMZ_TOURNEES_FILE = "abonnes_portes_analyste_tournee.kmz"

# Facteur multiplicateur sur le seuil nearest‐neighbor (pour tolérer léger décalage)
NN_THRESHOLD_FACTOR = 1.5

# Tampon minimal pour le convex hull (en degrés, ≃ 50 m à l’équateur)
HULL_BUFFER_DEGREES = 0.0005

# ------------------------------------------------------------------------------
#                          FONCTIONS UTILITAIRES
# ------------------------------------------------------------------------------
def distance_haversine(lat1, lon1, lat2, lon2):
    """
    Calcule la distance (en km) entre deux points GPS (lat1, lon1) et (lat2, lon2).
    """
    R = 6371.0  # rayon moyen de la Terre en km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def distance_haversine_array(lat0, lon0, lat_array, lon_array):
    """
    Version vectorisée : distance (en km) entre un point (lat0, lon0)
    et un tableau de points (lat_array, lon_array). Renvoie un np.ndarray.
    """
    R = 6371.0
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)
    lat_rad = np.radians(lat_array)
    lon_rad = np.radians(lon_array)

    dlat = lat_rad - lat0_rad
    dlon = lon_rad - lon0_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat0_rad) * np.cos(lat_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def clean_address(addr: str) -> str:
    """
    Nettoie l’adresse en :
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
      1) essai sur l’adresse brute + " France"
      2) essai sur l’adresse nettoyée + " France"
    Gère le code 429 avec un backoff exponentiel.
    Retourne (lat, lon) ou (None, None) si aucun résultat.
    """
    headers = {"User-Agent": USER_AGENT}
    variants = [address, clean_address(address)]
    for variant in variants:
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
                # Erreur réseau, pause puis sortie de cette variante
                time.sleep(1)
                break

            if r.status_code == 200:
                data = r.json()
                if data:
                    d = data[0]
                    return float(d["lat"]), float(d["lon"])
                # Pas de résultat (liste vide) → pause, puis on sort
                time.sleep(1)
                break

            elif r.status_code == 429:
                # Trop de requêtes → backoff
                time.sleep(backoff)
                backoff = min(backoff * 2, 5.0)
                continue

            else:
                # Autre code HTTP non géré → pause puis on sort
                time.sleep(1)
                break

    return None, None

def load_points_from_kmz(kmz_path: str):
    """
    Lit un fichier KMZ, extrait son KML, et renvoie :
      route_points_dict = { "NomTournée": np.array([[lat, lon], …]), … }
    """
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}

    # 1) Charger l’archive KMZ (ZIP) et identifier le .kml à l’intérieur
    with zipfile.ZipFile(kmz_path, 'r') as kmz:
        kml_files = [fn for fn in kmz.namelist() if fn.lower().endswith('.kml')]
        if not kml_files:
            raise FileNotFoundError(f"Aucun fichier .kml dans {kmz_path}")
        kml_name = kml_files[0]

        # 2) Parser le KML
        with kmz.open(kml_name, 'r') as f:
            tree = ET.parse(f)

    root = tree.getroot()
    route_points_dict = {}

    # 3) Pour chaque <Folder> dans le KML (chaque Folder = 1 tournée)
    for folder in root.findall('.//kml:Folder', ns):
        name_elem = folder.find('kml:name', ns)
        if name_elem is None or not name_elem.text:
            continue
        tourn_name = name_elem.text.strip()

        coords_list = []
        # Pour chaque <Placemark><Point><coordinates>lon,lat,alt</coordinates></Point></Placemark>
        for placemark in folder.findall('.//kml:Placemark', ns):
            coord_elem = placemark.find('.//kml:Point/kml:coordinates', ns)
            if coord_elem is not None and coord_elem.text:
                raw = coord_elem.text.strip()
                parts = raw.split(',')
                try:
                    lon = float(parts[0])
                    lat = float(parts[1])
                    coords_list.append((lat, lon))
                except ValueError:
                    pass

        if coords_list:
            route_points_dict[tourn_name] = np.array(coords_list, dtype=float)

    return route_points_dict

@st.cache_data
def load_tournees_with_nn_thresholds():
    """
    Charge tous les points historiques de chaque tournée depuis le KMZ,
    calcule pour chaque tournée :
      - son convex hull bufferisé
      - le seuil nearest‐neighbor (90ᵉ percentile)
    Renvoie :
      - route_points_dict : { nom_tournée: np.array([[lat,lon], …]), … }
      - thresholds_dict   : { nom_tournée: seuil_N_km }
      - hulls_dict        : { nom_tournée: shapely.Polygon.buffered }
    """
    # 1) On extrait tous les points depuis le KMZ
    route_points_dict = load_points_from_kmz(KMZ_TOURNEES_FILE)

    thresholds_dict = {}
    hulls_dict = {}

    # 2) Pour chaque tournée, on calcule le 90ᵉ percentile nearest‐neighbor et le convex hull + buffer
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
    st.title("Attribution Automatique des Tournées PACA")
    st.write(
        "1) Uploade ton fichier clients (Adresse, CP, Ville…)\n"
        "2) L’app géocode, compare aux tournées existantes (KMZ) et attribue la tournée la plus proche\n"
        "3) Tu peux ensuite télécharger le fichier enrichi au format .xlsx"
    )

    # 1) Upload du fichier client
    uploaded = st.file_uploader("Fichier Excel/CSV", type=["xlsx", "xls", "csv"])
    if not uploaded:
        return

    # 2) Détection automatique de la ligne d’en‐tête
    raw = pd.read_excel(uploaded, header=None)
    header_idx = 0
    for i, row in raw.iterrows():
        txt = " ".join(map(str, row.tolist())).lower()
        if any(k in txt for k in ("adresse", "cp", "codepostal", "ville")):
            header_idx = i
            break

    df = pd.read_excel(uploaded, header=header_idx)
    st.write("Colonnes détectées :", list(df.columns))

    # 3) Détection (ou sélection manuelle) des colonnes Adresse, CP, Ville
    addr_cols = [c for c in df.columns if any(w in c.lower() for w in ("adresse","voie","rue","route","chemin"))]
    cp_candidates = [c for c in df.columns if ("codepostal" in c.lower()) or (c.lower() == "cp")]
    ville_candidates = [c for c in df.columns if "ville" in c.lower()]

    # Si on n’a pas trouvé automatiquement, on demande à l’utilisateur de choisir dans un selectbox
    if not addr_cols:
        st.warning("Impossible de détecter automatiquement la colonne 'Adresse'. Choisissez‐la manuellement.")
        choice_addr = st.selectbox("Quelle colonne contient l’adresse ?", options=list(df.columns))
        addr_cols = [choice_addr]

    if not cp_candidates:
        st.warning("Impossible de détecter automatiquement la colonne 'Code Postal'. Choisissez‐la manuellement (ou laissez vide).")
        cp_candidates = ["" ] + list(df.columns)
        cp_col = st.selectbox("Quelle colonne contient le code postal ? (laisser vide si pas présente)", options=cp_candidates)
        cp_col = cp_col if cp_col != "" else None
    else:
        cp_col = cp_candidates[0]

    if not ville_candidates:
        st.warning("Impossible de détecter automatiquement la colonne 'Ville'. Choisissez‐la manuellement (ou laissez vide).")
        ville_candidates = ["" ] + list(df.columns)
        ville_col = st.selectbox("Quelle colonne contient la ville ? (laisser vide si pas présente)", options=ville_candidates)
        ville_col = ville_col if ville_col != "" else None
    else:
        ville_col = ville_candidates[0]

    st.write(f"→ Colonnes sélectionnées : Adresse(s)={addr_cols}, CodePostal={cp_col}, Ville={ville_col}")

    # 4) Construction du champ d’adresse complète (normalisée)
    df["_full_address"] = ""
    for c in addr_cols + ([cp_col] if cp_col else []) + ([ville_col] if ville_col else []):
        if c:
            df["_full_address"] += df[c].fillna("").astype(str) + " "
    df["_full_address"] = df["_full_address"].apply(lambda x: unidecode(x.strip()))

    # 5) Géocodage avec feedback visuel
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

    # Vérifier si on a au moins un point géocodé valablement
    num_valid = df[ ["Latitude","Longitude"] ].dropna().shape[0]
    if num_valid == 0:
        st.error("Aucune adresse n'a pu être géocodée. Vérifiez vos colonnes Adresse/CP/Ville ou la qualité des données.")
        return
    st.success(f"✅ Géocodage terminé ({num_valid}/{total} adresses valides)")

    # 6) Chargement des tournées historiques + calcul des seuils et hulls
    try:
        route_points_dict, thresholds_dict, hulls_dict = load_tournees_with_nn_thresholds()
    except Exception as e:
        st.error(f"Impossible de charger le KMZ « {KMZ_TOURNEES_FILE} » ou d’en extraire les tournées :\n {e}")
        return

    # Afficher combien de tournées on a extrait du KMZ
    st.write(f"📂 {len(route_points_dict)} tournées chargées depuis le KMZ.")
    if len(route_points_dict) == 0:
        st.error("Le KMZ ne contient aucune tournée. Vérifiez que vous avez bien uploadé « abonnes_portes_analyste_tournee.kmz » contenant les folders KML.")
        return

    # 7) Afficher à titre de diagnostic quelques tournées + leur nombre de points
    sample_routes = list(route_points_dict.items())[:5]
    st.write("Exemple des premières tournées extraites :")
    for rn, pts in sample_routes:
        st.write(f"   • {rn} → {pts.shape[0]} points historiques")

    # 8) Attribution prioritaire par CONVEX HULL tamponné, puis fallback NN
    st.write("🚚 Attribution des tournées (étape en cours)…")
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

        # 8.1) Si le client est dans le convex hull bufferisé d’une tournée, on l’affecte directement
        for route_name, hull_buf in hulls_dict.items():
            if hull_buf.contains(pt):
                choix = route_name
                break

        # 8.2) Sinon fallback nearest‐neighbor
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

    # 9) Export en .xlsx
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
