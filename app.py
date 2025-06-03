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

# ---------------------------------------------------------------------------
#                         FONCTIONS UTILITAIRES
# ---------------------------------------------------------------------------

def strip_ns(tag: str) -> str:
    """
    Supprime le namespace d'une balise XML.
    Ex. '{http://www.opengis.net/kml/2.2}Folder'  →  'Folder'
    """
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag

@st.cache_data(show_spinner=False)
def load_points_from_kml(kml_path: str) -> dict:
    """
    Lit un fichier KML non compressé et renvoie un dict :
      { 'NomTournée': np.array([[lat, lon], …]), … }.

    Structure attendue du KML (dans cet ordre) :
      <kml>
        <Document>
          <Folder>           ← Dépositaire (ex. "MARSEILLE")
            <name>MARSEILLE</name>
            <Folder>         ← Tournée (ex. "A001 - ALLAUCH 1")
              <name>A001 - ALLAUCH 1</name>
              …                ← sous-<Folder> éventuels contenant <Placemark>
                <Placemark>
                  <Point><coordinates>lon,lat,alt</coordinates></Point>
                </Placemark>
              …
            </Folder>
            <Folder>         ← Tournée suivante
              …
            </Folder>
          </Folder>
          <Folder>           ← Autre Dépositaire
            …
          </Folder>
        </Document>
      </kml>

    Pour chaque <Folder> de niveau “Tournée” (c’est‐à‐dire enfant direct d’un “Dépositaire”),
    on récupère tous les <Placemark> descendants (dans tous les sous-<Folder>),
    on extrait (lat,lon) et on stocke dans route_points[tourn_name].
    """
    tree = ET.parse(kml_path)
    root = tree.getroot()

    # Cherche l'élément <Document> (sinon on reste à la racine)
    doc_elem = None
    for child in root:
        if strip_ns(child.tag) == "Document":
            doc_elem = child
            break
    if doc_elem is None:
        doc_elem = root

    route_points = {}

    def extract_coords_from_placemark(pm_elem):
        """
        Lit un <Placemark> et renvoie (lat, lon) ou None
        (on cherche la première balise <coordinates> = "lon,lat,alt").
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

    # 1) Pour chaque “dépositaire” (Folder enfant de <Document>)
    for depos_elem in doc_elem:
        if strip_ns(depos_elem.tag) != "Folder":
            continue

        # 2) Pour chaque “tournée” (Folder enfant du dépositaire)
        for tourn_elem in depos_elem:
            if strip_ns(tourn_elem.tag) != "Folder":
                continue

            # Lire le <name> du folder “tournée”
            tourn_name = None
            for kid in tourn_elem:
                if strip_ns(kid.tag) == "name" and kid.text:
                    tourn_name = kid.text.strip()
                    break
            if not tourn_name:
                continue

            # 3) Extraire tous les Placemark descendants de cette tournée
            coords_list = []
            for pm in tourn_elem.iter():
                if strip_ns(pm.tag) == "Placemark":
                    coord = extract_coords_from_placemark(pm)
                    if coord:
                        coords_list.append(coord)

            if coords_list:
                route_points[tourn_name] = np.array(coords_list, dtype=float)

    return route_points


@st.cache_data(show_spinner=False)
def load_tournees_with_nn_thresholds(
    kml_file: str,
    default_threshold_km: float = 0.3
):
    """
    Charge les tournées depuis un KML (non compressé) :
      - route_points_dict : { 'NomTournée': np.array([[lat,lon], ...]), ... }
      - thresholds_dict   : { 'NomTournée': seuil_km, ... }
      - polygons_dict     : { 'NomTournée': shapely Polygon (ou MultiPolygon), ... }

    Pour chaque tournée :
      1) On extrait son nuage de points (lat, lon).
      2) On calcule la distance “nearest‐neighbor” (NN) de chaque point à son plus proche
         voisin (dans la même tournée) → on stocke dans df_ref["NN_dist_km"].
      3) Seuil = max(médiane(NN_dist_km) × 3, default_threshold_km).   ← **× 3 au lieu de × 2**
      4) On crée autour de chaque point un buffer de rayon (seuil km) converti en degrés
         (1 km ≃ 1 / 111 degrés). On fait l’union de tous ces buffers → corridor.
    Retourne (route_points_dict, thresholds_dict, polygons_dict).
    """
    route_points_dict = load_points_from_kml(kml_file)
    if not route_points_dict:
        return None, None, None

    # 1) Construire un DataFrame “plat” pour calcul NN
    rows = []
    for tourn_name, pts in route_points_dict.items():
        for (lat, lon) in pts:
            rows.append({"Tournée": tourn_name, "Latitude": lat, "Longitude": lon})
    df_ref = pd.DataFrame(rows)

    # 2) Calculer la distance NN pour chaque point
    def haversine_array(lat1, lon1, lat2_arr, lon2_arr):
        """
        Pour un point (lat1,lon1) et un array de points lat2_arr,lon2_arr,
        calcule la distance Haversine vers chacun, retourne un array de distances en km.
        """
        lat1r = radians(lat1)
        lon1r = radians(lon1)
        lat2r = np.radians(lat2_arr)
        lon2r = np.radians(lon2_arr)
        dlat = lat2r - lat1r
        dlon = lon2r - lon1r
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return 6371.0 * c  # rayon Terre ≃ 6371 km

    nn_distances = []
    for tourn_name, grp in df_ref.groupby("Tournée"):
        latitudes = grp["Latitude"].values
        longitudes = grp["Longitude"].values
        n_pts = len(latitudes)
        if n_pts <= 1:
            # Si un seul point, pas de voisin proche ⇒ inf
            nn_distances.extend([np.inf] * n_pts)
            continue

        for i in range(n_pts):
            lat_i = latitudes[i]
            lon_i = longitudes[i]
            dists = haversine_array(lat_i, lon_i, latitudes, longitudes)
            dists[i] = np.inf
            nn_distances.append(dists.min())

    df_ref["NN_dist_km"] = nn_distances

    # 3) Définir le seuil et construire le “corridor” (buffer union)
    thresholds = {}
    polygons = {}
    for tourn_name, grp in df_ref.groupby("Tournée"):
        median_nn = grp["NN_dist_km"].replace(np.inf, 0).median()
        seuil = float(max(median_nn * 3, default_threshold_km))
        thresholds[tourn_name] = seuil

        # Conversion km → degrés : 1° ≃ 111 km
        radius_degrees = seuil / 111.0

        # 4) Pour chaque point, créer un disk (lon,lat).buffer(radius_degrees)
        pts = route_points_dict[tourn_name]
        buffers = []
        for (lat, lon) in pts:
            p = Point(lon, lat)
            buffers.append(p.buffer(radius_degrees))
        union_poly = unary_union(buffers)
        polygons[tourn_name] = union_poly

    return route_points_dict, thresholds, polygons


@st.cache_data(show_spinner=False)
def geocode(address: str):
    """
    Géocode une adresse via Nominatim (OSM) avec backoff sur 429.
    On tente l’adresse brute puis l’adresse “nettoyée” (clean_address).
    Retourne (lat, lon) ou (None, None).
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
                time.sleep(1)
                break

            elif resp.status_code == 429:
                # Trop de requêtes : backoff exponentiel jusqu’à 5 s
                time.sleep(backoff)
                backoff = min(backoff * 2, 5.0)
                continue

            else:
                time.sleep(1)
                break

    return None, None


def haversine_km(lat1, lon1, lat2, lon2):
    """
    Calcule la distance Haversine en km entre (lat1,lon1) et (lat2,lon2).
    """
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


# ---------------------------------------------------------------------------
#                               FONCTION PRINCIPALE
# ---------------------------------------------------------------------------

def main():
    st.title("Attribution Automatique des Tournées PACA")
    st.write("""
    1) Uploadez votre fichier clients (Adresse, CP, Ville…)  
    2) L’app géocode chaque client et l’associe à la tournée la plus proche  
    3) Téléchargez le résultat enrichi (.xlsx)  
    """)

    # --- 1) Fichier clients ---
    uploaded = st.file_uploader("Fichier client (Excel/CSV)", type=["xlsx", "xls", "csv"])
    if not uploaded:
        return

    # 2) Détection de la ligne d'en-tête (contenant “adresse”, “cp” ou “ville”)
    raw = pd.read_excel(uploaded, header=None)
    header_idx = 0
    for i, row in raw.iterrows():
        txt = " ".join(map(str, row.tolist())).lower()
        if any(k in txt for k in ("adresse", "cp", "codepostal", "ville")):
            header_idx = i
            break

    df_clients = pd.read_excel(uploaded, header=header_idx)
    st.write("Colonnes détectées :", list(df_clients.columns))

    # 3) Détection auto des colonnes “Adresse”, “Code Postal”, “Ville”
    addr_cols = [c for c in df_clients.columns if any(w in c.lower() for w in ("adresse","voie","rue","route","chemin"))]
    cp_candidates = [c for c in df_clients.columns if "codepostal" in c.lower() or c.lower() == "cp"]
    ville_candidates = [c for c in df_clients.columns if "ville" in c.lower()]

    if not addr_cols:
        st.warning("Impossible de détecter la colonne 'Adresse'. Choisissez-la manuellement.")
        choix_addr = st.selectbox("Colonne Adresse ?", options=list(df_clients.columns))
        addr_cols = [choix_addr]

    if not cp_candidates:
        st.warning("Impossible de détecter la colonne 'Code Postal'. Sélectionnez-la ou laissez vide.")
        choix_cp = st.selectbox("Colonne Code Postal ? (laisser vide si pas présent)", [None] + list(df_clients.columns))
        cp_col = choix_cp if choix_cp else None
    else:
        cp_col = cp_candidates[0]

    if not ville_candidates:
        st.warning("Impossible de détecter la colonne 'Ville'. Sélectionnez-la ou laissez vide.")
        choix_ville = st.selectbox("Colonne Ville ? (laisser vide si pas présent)", [None] + list(df_clients.columns))
        ville_col = choix_ville if choix_ville else None
    else:
        ville_col = ville_candidates[0]

    st.write(f"→ Colonnes sélectionnées : Adresse={addr_cols}, CP={cp_col}, Ville={ville_col}")

    # 4) Concaténation en “_full_address”
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

    # 7) Attribution des tournées aux clients
    st.write("🚚 Attribution des tournées en cours…")
    progress_attr = st.progress(0)
    attribs = []

    for i, row in enumerate(df_clients.itertuples()):
        latc = getattr(row, "Latitude")
        lonc = getattr(row, "Longitude")
        if pd.isna(latc) or pd.isna(lonc):
            attribs.append("")  # pas de coordonnées ⇒ pas d’attribution
            progress_attr.progress((i + 1) / total)
            continue

        pt = Point(lonc, latc)
        choix = ""

        # 7.1) Vérifier si le client est DANS le corridor (polygone) d’une tournée
        for tourn_name, poly in polygons_dict.items():
            if poly.contains(pt):
                choix = tourn_name
                break

        # 7.2) Sinon, fallback nearest‐neighbor
        if choix == "":
            best_tour = ""
            best_dist = float("inf")
            for tourn_name, pts in route_points_dict.items():
                arr = np.array(pts)  # shape (N,2) avec (lat, lon)
                # calcul Haversine entre {latc,lonc} et chaque (lat,lon) de la tournée
                dists_km = np.vectorize(lambda la, lo: haversine_km(latc, lonc, la, lo))(arr[:,0], arr[:,1])
                dmin = float(dists_km.min())
                if dmin < best_dist:
                    best_dist = dmin
                    best_tour = tourn_name

            seuil = thresholds_dict.get(best_tour, 0.0)
            if best_dist <= seuil:
                choix = best_tour
            else:
                choix = ""

        attribs.append(choix)
        progress_attr.progress((i + 1) / total)

    df_clients["Tournée attribuée"] = attribs
    st.success("✅ Attribution des tournées terminée.")

    # 8) Export du fichier enrichi en .xlsx
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
