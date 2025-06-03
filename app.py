import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import io
import xml.etree.ElementTree as ET
from math import radians, sin, cos, sqrt, atan2
from shapely.geometry import Point, MultiPoint

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
    Lecture d'un KML non compressé et extraction des tournées.
    La structure attendue :
      <kml>
        <Document>
          <Folder>          ← Dépositaire (ex. "MARSEILLE")
            <name>...</name>
            <Folder>        ← Tournée (ex. "A001 - ALLAUCH 1")
              <name>...</name>
              ... (Plusieurs sous-<Folder> → Placemark)
            </Folder>
            <Folder>        ← Autre tournée
              ...
            </Folder>
          </Folder>
          <Folder>          ← Autre Dépositaire
            ...
          </Folder>
        </Document>
      </kml>

    Pour chaque `<Folder>` de niveau "tournée" (i.e. enfant direct d'un dépositaire),
    on récupère **tous** ses <Placemark> descendants (dans les sous-sous-dossiers),
    puis on stocke dans route_points[tournée] = np.array([[lat, lon], ...]).
    """
    tree = ET.parse(kml_path)
    root = tree.getroot()

    # Cherche l'élément <Document>
    doc_elem = None
    for child in root:
        if strip_ns(child.tag) == "Document":
            doc_elem = child
            break
    if doc_elem is None:
        # Dans le doute, on prend la racine si pas de Document explicite
        doc_elem = root

    route_points = {}

    def extract_coords_from_placemark(pm_elem):
        """
        Lit un <Placemark> et renvoie (lat, lon) ou None si problème.
        On cherche la première balise <coordinates> : "lon,lat,alt"
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

    # Parcours des dépositaire → tournée
    for depos_elem in doc_elem:
        if strip_ns(depos_elem.tag) != "Folder":
            continue
        # depositeur_name = nom (ex. "MARSEILLE"), pas utilisé directement ici
        for tourn_elem in depos_elem:
            if strip_ns(tourn_elem.tag) != "Folder":
                continue
            # On récupère le <name> du folder "tournée"
            tourn_name = None
            for kid in tourn_elem:
                if strip_ns(kid.tag) == "name" and kid.text:
                    tourn_name = kid.text.strip()
                    break
            if not tourn_name:
                continue

            # Récupère tous les Placemark descendants de ce tourn_elem
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
    default_threshold_km: float = 0.15
):
    """
    Charge les tournées depuis le KML (non compressé) :
      - route_points_dict : { 'NomTournée': np.array([[lat,lon], ...]), ... }
      - thresholds_dict   : { 'NomTournée': seuil_km, ... }
      - hulls_dict        : { 'NomTournée': convex_hull (Shapely), ... }

    Seuil calculé comme médiane des distances NN × 2 ou minimum default_threshold_km.
    """
    route_points_dict = load_points_from_kml(kml_file)
    if not route_points_dict:
        return None, None, None

    # Construire un DataFrame plat pour calcul des plus proches voisins
    rows = []
    for tourn_name, pts in route_points_dict.items():
        for lat, lon in pts:
            rows.append({"Tournée": tourn_name, "Latitude": lat, "Longitude": lon})
    df_ref = pd.DataFrame(rows)

    # Calcul manuel des plus proches voisins (sans sklearn)
    # Pour chaque point, on calcule la distance haversine à tous les autres de la même tournée,
    # on prend la plus petite NON nulle → c'est la distance au "véritable" NN.
    nn_distances = []
    for tourn_name, grp in df_ref.groupby("Tournée"):
        # Extraire les lat/lon de cette tournée
        latitudes = grp["Latitude"].values
        longitudes = grp["Longitude"].values
        n = len(latitudes)
        if n <= 1:
            # Si un seul point, on considère NN = infinite
            nn_distances.extend([np.inf] * n)
            continue

        # Calcul pairwise distances pour cette tournée
        # On va calculer la matrice NxN des distances Haversine
        arr_lats = latitudes.reshape((n, 1))
        arr_lons = longitudes.reshape((n, 1))

        # Fonction pour appliquer haversine sur vecteurs
        def haversine_array(lat1, lon1, lat2arr, lon2arr):
            # lat1, lon1 scalaires, lat2arr, lon2arr vecteurs
            dlat = np.radians(lat2arr - lat1)
            dlon = np.radians(lon2arr - lon1)
            a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * \
                np.cos(np.radians(lat2arr)) * np.sin(dlon / 2) ** 2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            return 6371.0 * c  # rayon terre en km

        # Pour chaque i, calculer distance au plus proche autre point j != i
        for i in range(n):
            lat_i = latitudes[i]
            lon_i = longitudes[i]
            # calcul à toute la liste
            dists = haversine_array(lat_i, lon_i, latitudes, longitudes)
            dists[i] = np.inf  # ignorer soi-même
            nn_distances.append(dists.min())

    # On fixe la colonne NN_dist_km
    df_ref["NN_dist_km"] = nn_distances

    # Calcul des seuils et des convex hulls
    thresholds = {}
    hulls = {}
    for tourn_name, grp in df_ref.groupby("Tournée"):
        median_nn = grp["NN_dist_km"].replace(np.inf, 0).median()
        seuil = float(max(median_nn * 2, default_threshold_km))
        thresholds[tourn_name] = seuil

        # Construire un convex hull shapely
        pts = route_points_dict[tourn_name]
        shapely_pts = [Point(lon, lat) for lat, lon in pts]
        hull = MultiPoint(shapely_pts).convex_hull
        hulls[tourn_name] = hull

    return route_points_dict, thresholds, hulls


@st.cache_data(show_spinner=False)
def geocode(address: str):
    """
    Géocode via Nominatim (OSM) avec backoff en cas d'erreur 429.
    On teste d'abord l'adresse brute, puis l'adresse "nettoyée".
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
                # Trop de requêtes : on attend un peu puis on réessaye (jusqu'à 5 secondes)
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

    # ------------- Uploader le fichier "clients" -----------------
    uploaded = st.file_uploader("Fichier client (Excel/CSV)", type=["xlsx", "xls", "csv"])
    if not uploaded:
        return

    # 1) Scanner l'en-tête pour trouver la première ligne où apparaissent "adresse", "CP", "ville"
    raw = pd.read_excel(uploaded, header=None)
    header_idx = 0
    for i, row in raw.iterrows():
        txt = " ".join(map(str, row.tolist())).lower()
        if any(k in txt for k in ("adresse", "cp", "codepostal", "ville")):
            header_idx = i
            break

    df_clients = pd.read_excel(uploaded, header=header_idx)
    st.write("Colonnes détectées :", list(df_clients.columns))

    # 2) Détection automatique des colonnes "Adresse", "Code postal", "Ville"
    addr_cols = [c for c in df_clients.columns if any(w in c.lower() for w in ("adresse", "voie", "rue", "route", "chemin"))]
    cp_candidates = [c for c in df_clients.columns if "codepostal" in c.lower() or c.lower() == "cp"]
    ville_candidates = [c for c in df_clients.columns if "ville" in c.lower()]

    if not addr_cols:
        st.warning("Impossible de détecter automatiquement la colonne 'Adresse'. Choisissez-la manuellement.")
        choix_addr = st.selectbox("Colonne Adresse ?", options=list(df_clients.columns))
        addr_cols = [choix_addr]

    if not cp_candidates:
        st.warning("Impossible de détecter la colonne 'Code Postal'. Sélectionnez ou laissez vide.")
        choix_cp = st.selectbox("Colonne Code Postal ? (laisser vide si pas présent)", [None] + list(df_clients.columns))
        cp_col = choix_cp if choix_cp else None
    else:
        cp_col = cp_candidates[0]

    if not ville_candidates:
        st.warning("Impossible de détecter la colonne 'Ville'. Sélectionnez ou laissez vide.")
        choix_ville = st.selectbox("Colonne Ville ? (laisser vide si pas présent)", [None] + list(df_clients.columns))
        ville_col = choix_ville if choix_ville else None
    else:
        ville_col = ville_candidates[0]

    st.write(f"→ Colonnes sélectionnées : Adresse={addr_cols}, CP={cp_col}, Ville={ville_col}")

    # 3) Construction du champ d'adresse complète
    df_clients["_full_address"] = ""
    for c in addr_cols + ([cp_col] if cp_col else []) + ([ville_col] if ville_col else []):
        if c:
            df_clients["_full_address"] += df_clients[c].fillna("").astype(str) + " "
    df_clients["_full_address"] = df_clients["_full_address"].str.strip()

    # 4) Géocodage
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
        st.error("❌ Aucune adresse n’a pu être géocodée. Vérifiez les colonnes Adresse/CP/Ville.")
        return
    st.success(f"✅ Géocodage terminé ({n_valid}/{total} adresses valides).")

    # 5) Chargement des tournées depuis le KML
    st.write("📂 Extraction des tournées depuis le fichier KML…")
    KML_TOURNEES_FILE = "abonnes_portes_analyste_tournee.kml"
    route_points_dict, thresholds_dict, hulls_dict = load_tournees_with_nn_thresholds(KML_TOURNEES_FILE)
    if route_points_dict is None:
        st.error("❌ Le KML ne contient aucune tournée valide. Vérifiez la structure (Folder → Placemark).")
        return

    n_tournees = len(route_points_dict)
    st.success(f"🗂 {n_tournees} tournées extraites depuis le KML.")

    # 6) Attribution des tournées
    st.write("🚚 Attribution des tournées en cours…")
    progress_attr = st.progress(0)
    attribs = []
    for i, row in enumerate(df_clients.itertuples()):
        latc = getattr(row, "Latitude")
        lonc = getattr(row, "Longitude")
        if pd.isna(latc) or pd.isna(lonc):
            attribs.append("")  # pas de coordonnées
            progress_attr.progress((i + 1) / total)
            continue

        pt = Point(lonc, latc)
        choix = ""

        # 6.1) Recherche par convex hull
        for tourn_name, hull in hulls_dict.items():
            if hull.contains(pt):
                choix = tourn_name
                break

        # 6.2) Si pas trouvé via hull, fallback nearest-neighbor
        if choix == "":
            best_tour = ""
            best_dist = float("inf")
            for tourn_name, pts in route_points_dict.items():
                arr = np.array(pts)
                # calcul des distances "haversine" en km vers TOUS points de la tournée
                dists = np.vectorize(lambda la, lo: haversine_km(latc, lonc, la, lo))(arr[:,0], arr[:,1])
                dmin = float(dists.min())
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

    # 7) Export en .xlsx
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
