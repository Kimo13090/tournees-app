# app.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import io
import xml.etree.ElementTree as ET
from math import radians, sin, cos, sqrt, atan2
from shapely.geometry import Point
from shapely.ops import unary_union

# --------------------------------------------------------------------
#                       FONCTIONS UTILES
# --------------------------------------------------------------------

def normalize_header_cell(cell_value: str) -> str:
    """
    Prend un intitulé de colonne (par ex. 'Ville :', 'Adresse (1)', 'CP ;') et renvoie
    une version “normalisée” en minuscules, sans espaces ni signes de ponctuation.
    Exemples :
      'Ville :'        → 'ville'
      '  Code Postal ' → 'codepostal'
      'AdrESSe(1)'     → 'adresse1'
      'CP ;'           → 'cp'
    """
    if cell_value is None:
        return ""
    s = str(cell_value).strip().lower()
    # On ne garde que les caractères alphanumériques (lettres & chiffres)
    normalized = "".join(ch for ch in s if ch.isalnum())
    return normalized


def detect_header_row(df_raw: pd.DataFrame) -> int:
    """
    Recherche la ligne d'en-tête dans df_raw (qui a été lu avec header=None).
    On normalise chaque cellule de chaque ligne, puis on regarde si dans cette ligne
    on trouve au moins l’un des mots-clés :
      - 'adresse'   → mot-clé pour la colonne Adresse
      - 'codepostal' ou 'cp'   → mot-clés pour la colonne Code Postal
      - 'ville'     → mot-clé pour la colonne Ville
    Si on trouve un de ces mots-clés dans la ligne, on renvoie son index.
    Sinon on renvoie 0 par défaut.
    """
    keywords = {"adresse", "codepostal", "cp", "ville"}
    for i, row in df_raw.iterrows():
        normalized_cells = [normalize_header_cell(cell) for cell in row.tolist()]
        for nc in normalized_cells:
            for kw in keywords:
                if kw in nc:
                    return i
    return 0



@st.cache_data(show_spinner=False)
def load_points_from_kml(kml_path: str) -> dict:
    """
    Lit un KML non compressé et retourne un dictionnaire :
      { 'NomTournée': np.array([[lat, lon], …]), … }.

    On s'attend à une structure comme :
      <kml>
        <Document>
          <Folder>               ← Dépositaire (ex. 'MARSEILLE')
            <name>MARSEILLE</name>
            <Folder>             ← Tournée (ex. 'A001 - ALLAUCH 1')
              <name>A001 - ALLAUCH 1</name>
              <Folder> … </Folder>  ← Sous-dossier (typiquement 'Autre' ou 'VILLES', etc.)
                <Placemark>
                  <Point>
                    <coordinates>5.47443,43.35647,0</coordinates>
                  </Point>
                </Placemark>
                …
              </Folder>
            </Folder>
            <Folder> … </Folder>  ← Autre tournée
          </Folder>
          <Folder> … </Folder>    ← Autre Dépositaire (ex. 'AIX')
        </Document>
      </kml>

    Pour chaque Folder de niveau “Tournée” (fils direct d’un Dépositaire), on récupère
    tous les Placemark contenus, quelles que soient la profondeur des sous-Folder.
    """
    def strip_ns(tag: str) -> str:
        # Supprime le namespace '{…}' si présent
        return tag.split("}", 1)[1] if "}" in tag else tag

    def extract_coord_from_placemark(pm_elem):
        """
        Lit un Placemark, cherche la première balise <coordinates> et retourne (lat, lon) ou None.
        <coordinates> est sous la forme "lon,lat,altitude".
        """
        for node in pm_elem.iter():
            if strip_ns(node.tag) == "coordinates" and node.text:
                raw = node.text.strip()
                parts = raw.split(",")
                try:
                    lon = float(parts[0])
                    lat = float(parts[1])
                    return lat, lon
                except (ValueError, IndexError):
                    return None
        return None

    tree = ET.parse(kml_path)
    root = tree.getroot()

    # Trouver l’élément <Document> (sinon, on part de root directement)
    doc_elem = None
    for child in root:
        if strip_ns(child.tag) == "Document":
            doc_elem = child
            break
    if doc_elem is None:
        doc_elem = root

    route_points = {}

    # Parcours de chaque “Dépositaire” (Folder enfant du Document)
    for depos_elem in doc_elem:
        if strip_ns(depos_elem.tag) != "Folder":
            continue

        # Parcours de chaque “Tournée” (Folder enfant du Dépositaire)
        for tourn_elem in depos_elem:
            if strip_ns(tourn_elem.tag) != "Folder":
                continue

            # Extraire le nom de la tournée
            tourn_name = None
            for kid in tourn_elem:
                if strip_ns(kid.tag) == "name" and kid.text:
                    tourn_name = kid.text.strip()
                    break
            if not tourn_name:
                continue

            # Récupérer tous les Placemark contenus dans cette tournée
            coords_list = []
            for pm in tourn_elem.iter():
                if strip_ns(pm.tag) == "Placemark":
                    coord = extract_coord_from_placemark(pm)
                    if coord:
                        coords_list.append(coord)
            if coords_list:
                route_points[tourn_name] = np.array(coords_list, dtype=float)

    return route_points


@st.cache_data(show_spinner=False)
def load_tournees_with_nn_thresholds(
    kml_file: str,
    default_threshold_km: float = 0.15
):
    """
    Charge un fichier KML non compressé de tournées, retourne :
      - route_points_dict : { 'Tournée': np.array([[lat,lon], ...]), … }
      - thresholds_dict   : { 'Tournée': seuil_km, … }
      - polygons_dict     : { 'Tournée': shapely Polygon (ou MultiPolygon), … }

    Pour chaque Tournée :
      1) On récupère le nuage de points (lat,lon).
      2) On calcule, pour chaque point, sa plus petite distance (Haversine) à un autre
         point de la même tournée (NN = nearest-neighbor).
      3) On prend médiane(NN) × 2 comme “seuil” (au minimum default_threshold_km).
      4) On construit un buffer vis-à-vis de chaque point (rayon = seuil en degrés ≃ seuil/111)
         puis on fait l'union des cercles obtenus pour former un “corridor” polygonal.
    """
    route_points_dict = load_points_from_kml(kml_file)
    if not route_points_dict:
        return None, None, None

    # 1) Mettre toutes les coordonnées dans un DataFrame plat
    rows = []
    for tourn_name, pts in route_points_dict.items():
        for lat, lon in pts:
            rows.append({"Tournée": tourn_name, "Latitude": lat, "Longitude": lon})
    df_ref = pd.DataFrame(rows)

    # 2) Calculer la distance Haversine entre chaque point i et tous les points j de la même tournée
    def haversine_array(lat1, lon1, lat_arr, lon_arr):
        lat1r = radians(lat1)
        lon1r = radians(lon1)
        lat2r = np.radians(lat_arr)
        lon2r = np.radians(lon_arr)
        dlat = lat2r - lat1r
        dlon = lon2r - lon1r
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return 6371.0 * c  # Rayon moyen Terre ≃ 6371 km

    nn_distances = []
    for tourn_name, grp in df_ref.groupby("Tournée"):
        latitudes = grp["Latitude"].values
        longitudes = grp["Longitude"].values
        n_pts = len(latitudes)
        if n_pts <= 1:
            nn_distances.extend([np.inf] * n_pts)
            continue

        for i in range(n_pts):
            lat_i = latitudes[i]
            lon_i = longitudes[i]
            dists = haversine_array(lat_i, lon_i, latitudes, longitudes)
            dists[i] = np.inf  # ignorer la distance à soi-même
            nn_distances.append(dists.min())

    df_ref["NN_dist_km"] = nn_distances

    # 3) Pour chaque tournée, on détermine un seuil = max(médiane(NN)*2, default_threshold_km)
    thresholds = {}
    polygons = {}
    for tourn_name, grp in df_ref.groupby("Tournée"):
        median_nn = grp["NN_dist_km"].replace(np.inf, 0).median()
        seuil = float(max(median_nn * 2, default_threshold_km))
        thresholds[tourn_name] = seuil

        # 4) Créer un buffer autour de chacun des points pour obtenir un “corridor”
        pts = route_points_dict[tourn_name]
        radius_degrees = seuil / 111.0  # approximativement 1° ≃ 111 km

        buffers = []
        for (lat, lon) in pts:
            shp_pt = Point(lon, lat)  # shapely prend l’ordre (x=lon, y=lat)
            circ = shp_pt.buffer(radius_degrees)
            buffers.append(circ)

        union_poly = unary_union(buffers)
        polygons[tourn_name] = union_poly

    return route_points_dict, thresholds, polygons


@st.cache_data(show_spinner=False)
def geocode(address: str):
    """
    Géocode une adresse via l’API Nominatim d’OpenStreetMap, avec backoff sur code 429.
    On teste l’adresse brute, puis une version “nettoyée” (clean_address).
    """
    USER_AGENT = "TourneeLocator/1.0 (contact@votredomaine.com)"

    def clean_address(addr2: str) -> str:
        tokens = addr2.split()
        cleaned = []
        prev = None
        for t in tokens:
            tl = t.lower().strip(".,")
            if tl in ("bd", "bld", "boul"):
                t2 = "boulevard"
            elif tl in ("av", "av.", "aven"):
                t2 = "avenue"
            elif tl in ("res", "res."):
                t2 = "résidence"
            else:
                t2 = t
            if t2.lower() != prev:
                cleaned.append(t2)
                prev = t2.lower()
        return " ".join(cleaned)

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
                    try:
                        return float(d0["lat"]), float(d0["lon"])
                    except Exception:
                        return None, None
                # pas de résultat, on sort
                time.sleep(1)
                break

            elif resp.status_code == 429:
                # Trop de requêtes → on backoff
                time.sleep(backoff)
                backoff = min(backoff * 2, 5.0)
                continue

            else:
                time.sleep(1)
                break

    return None, None


def haversine_km(lat1, lon1, lat2, lon2):
    """
    Calcule la distance Haversine (en km) entre (lat1,lon1) et (lat2,lon2).
    """
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


# --------------------------------------------------------------------
#                               MAIN
# --------------------------------------------------------------------

def main():
    st.title("Attribution Automatique des Tournées PACA")
    st.write("""
    1. Uploadez votre fichier clients (Excel/CSV) contenant au moins :
         - une colonne « Adresse » (ou « Rue », « Chemin », …)  
         - une colonne « CP » ou « Code Postal »  
         - une colonne « Ville »  
    2. L’app géocode chaque adresse et associe chaque client à la tournée la plus proche  
    3. Téléchargez le fichier enrichi (.xlsx)  
    """)

    # --- 1) Uploader le fichier client ---
    uploaded = st.file_uploader("Fichier client (Excel/CSV)", type=["xlsx", "xls", "csv"])
    if not uploaded:
        return

    # 2) Lire d'abord sans en-tête pour détecter la vraie ligne d'en-tête
    try:
        df_raw = pd.read_excel(uploaded, header=None)
    except Exception:
        st.error("❌ Impossible de lire votre fichier. Vérifiez qu’il est bien au format Excel ou CSV.")
        return

    header_idx = detect_header_row(df_raw)
    df_clients = pd.read_excel(uploaded, header=header_idx)

    st.write("Colonnes détectées :", list(df_clients.columns))

    # 3) Normaliser les noms de colonnes pour retrouver automatiquement Adresse/CP/Ville
    cols_norm = {col: normalize_header_cell(col) for col in df_clients.columns}

    # 3a) Chercher la/les colonne(s) « Adresse »
    addr_cols = []
    for col, norm in cols_norm.items():
        if (
            ("adresse" in norm)
            or ("adr" in norm and not norm == "cp")  # éviter de prendre "cp" si présent
            or ("rue" in norm)
            or ("voie" in norm)
            or ("chemin" in norm)
        ):
            addr_cols.append(col)

    # 3b) Chercher la colonne « Code Postal »
    cp_col = None
    for col, norm in cols_norm.items():
        if norm == "cp" or "codepostal" in norm:
            cp_col = col
            break

    # 3c) Chercher la colonne « Ville »
    ville_col = None
    for col, norm in cols_norm.items():
        if "ville" in norm:
            ville_col = col
            break

    # Si on n’a pas trouvé automatiquement, on demande à l’utilisateur
    if not addr_cols:
        st.warning("Impossible de détecter automatiquement la/les colonne(s) « Adresse ». Choisissez manuellement.")
        choix_addr = st.selectbox("Colonne Adresse ?", options=list(df_clients.columns))
        addr_cols = [choix_addr]

    if not cp_col:
        st.warning("Impossible de détecter automatiquement la colonne « Code Postal ». Sélectionnez-la ou laissez vide.")
        choix_cp = st.selectbox("Colonne Code Postal ? (laisser vide si pas présent)", [None] + list(df_clients.columns))
        cp_col = choix_cp

    if not ville_col:
        st.warning("Impossible de détecter automatiquement la colonne « Ville ». Sélectionnez-la ou laissez vide.")
        choix_ville = st.selectbox("Colonne Ville ? (laisser vide si pas présent)", [None] + list(df_clients.columns))
        ville_col = choix_ville

    st.write(f"→ Colonnes sélectionnées : Adresse={addr_cols}, Code Postal={cp_col}, Ville={ville_col}")

    # 4) Construction d’une unique colonne « _full_address »
    df_clients["_full_address"] = ""
    for c in addr_cols + ([cp_col] if cp_col else []) + ([ville_col] if ville_col else []):
        if c:
            df_clients["_full_address"] += df_clients[c].fillna("").astype(str) + " "
    df_clients["_full_address"] = df_clients["_full_address"].str.strip()

    # 5) Géocodage
    total = len(df_clients)
    st.write(f"🔍 Géocodage de {total} adresses…")
    progress_geo = st.progress(0)
    lats, lons = [], []
    for i, addr in enumerate(df_clients["_full_address"]):
        lat, lon = geocode(addr)
        lats.append(lat)
        lons.append(lon)
        progress_geo.progress((i + 1) / total)
    df_clients["Latitude"] = lats
    df_clients["Longitude"] = lons

    n_valid = df_clients[["Latitude", "Longitude"]].dropna().shape[0]
    if n_valid == 0:
        st.error("❌ Aucune adresse n’a pu être géocodée. Vérifiez vos colonnes Adresse/CP/Ville.")
        return
    st.success(f"✅ Géocodage terminé : {n_valid}/{total} adresses valides.")

    # 6) Chargement des tournées depuis le KML
    st.write("📂 Extraction des tournées depuis le fichier KML…")
    KML_TOURNEES_FILE = "abonnes_portes_analyste_tournee.kml"
    route_points_dict, thresholds_dict, polygons_dict = load_tournees_with_nn_thresholds(KML_TOURNEES_FILE)
    if route_points_dict is None:
        st.error("❌ Le KML ne contient aucune tournée valide. Vérifiez la structure (Folder → Placemark).")
        return

    n_tournees = len(route_points_dict)
    st.success(f"🗂 {n_tournees} tournées extraites depuis le KML.")

    # 7) Attribution des tournées
    st.write("🚚 Attribution des tournées en cours…")
    progress_attr = st.progress(0)
    attribs = []

    for i, row in enumerate(df_clients.itertuples()):
        latc = getattr(row, "Latitude")
        lonc = getattr(row, "Longitude")
        if pd.isna(latc) or pd.isna(lonc):
            attribs.append("")      # Pas de coords : on laisse vide
            progress_attr.progress((i + 1) / total)
            continue

        pt = Point(lonc, latc)
        choix = ""

        # 7.1) D'abord, on regarde si le point se trouve DANS le corridor de l’une des tournées
        for tourn_name, poly in polygons_dict.items():
            if poly.contains(pt):
                choix = tourn_name
                break

        # 7.2) Si aucune tournée n’a contenu le point, on fait nearest‐neighbor
        if choix == "":
            best_tour = ""
            best_dist = float("inf")
            for tourn_name, pts in route_points_dict.items():
                arr = np.array(pts)   # (N,2) avec (lat, lon)
                # Calcul vectorisé des distances Haversine
                dists = np.vectorize(lambda la, lo: haversine_km(latc, lonc, la, lo))(arr[:,0], arr[:,1])
                dmin = float(dists.min())
                if dmin < best_dist:
                    best_dist = dmin
                    best_tour = tourn_name

            seuil = thresholds_dict.get(best_tour, 0.0)
            if best_dist <= seuil:
                choix = best_tour
            else:
                choix = ""  # Trop loin, on ne l’attribue à aucune tournée

        attribs.append(choix)
        progress_attr.progress((i + 1) / total)

    df_clients["Tournée attribuée"] = attribs
    st.success("✅ Attribution des tournées terminée.")

    # 8) Export final
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_clients.to_excel(writer, index=False)
    st.download_button(
        "📥 Télécharger le fichier enrichi (.xlsx)",
        buffer.getvalue(),
        file_name="clients_tournees_enrichi.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


if __name__ == "__main__":
    main()
