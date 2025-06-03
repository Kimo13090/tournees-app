import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import io
import xml.etree.ElementTree as ET
from math import radians, sin, cos, sqrt, atan2
from shapely.geometry import Point, MultiPoint
from sklearn.neighbors import NearestNeighbors

# -------------------------------------------------------------------
#                            FONCTIONS UTILITAIRES
# -------------------------------------------------------------------

def strip_ns(tag: str) -> str:
    """
    Supprime un éventuel namespace d'une balise XML.
    Ex. '{http://www.opengis.net/kml/2.2}Folder' → 'Folder'.
    """
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag

@st.cache_data(show_spinner=False)
def load_points_from_kml(kml_filebuffer):
    """
    Lit un fichier KML non compressé (fichier-like ou chemin) et renvoie un dict :
        { 'NomTournée': np.array([[lat, lon], …]), … }.
    Parcourt récursivement chaque <Folder> pour extraire ses <Placemark><Point><coordinates>.
    Gère les dossiers imbriqués (ex. un sous-folder 'Autre' contenant les placemarks).
    """
    # Si kml_filebuffer est un buffer en mémoire (BytesIO), on l'ouvre directement.
    # Sinon on suppose que c'est un chemin local.
    try:
        tree = ET.parse(kml_filebuffer)
    except Exception as e:
        # Si échec de parse, retourne vide
        return {}

    root = tree.getroot()
    route_points = {}

    def extrait_placemarks(placemarks_list):
        """
        À partir d'une liste d'éléments <Placemark>, retourne une liste [(lat, lon), …].
        """
        pts = []
        for pm in placemarks_list:
            coord_elem = None
            for node in pm.iter():
                if strip_ns(node.tag) == "coordinates" and node.text:
                    coord_elem = node
                    break
            if coord_elem is None:
                continue
            raw = coord_elem.text.strip()
            # Format standard KML : "lon,lat[,alt]"
            parts = raw.split(",")
            try:
                lon = float(parts[0])
                lat = float(parts[1])
                pts.append((lat, lon))
            except (ValueError, IndexError):
                continue
        return pts

    def process_folder(folder_elem):
        """
        Traite un élément <Folder> pour en extraire une tournée :
        1) Si <Folder> a des <Placemark> directs, on lit <name> du folder et on extrait 
           leurs coordonnées.
        2) Sinon, si il a exactement un sous-<Folder> avec des <Placemark>, 
           on rattache ces placemarks au nom du folder parent.
        3) Sinon, on descend récursivement dans chaque sous-<Folder>.
        """
        # 1) Placemark directs ?
        placemarks_directs = [c for c in folder_elem if strip_ns(c.tag) == "Placemark"]
        if placemarks_directs:
            tourn_name = None
            for child in folder_elem:
                if strip_ns(child.tag) == "name" and child.text:
                    tourn_name = child.text.strip()
                    break
            if tourn_name:
                coords = extrait_placemarks(placemarks_directs)
                if coords:
                    route_points[tourn_name] = np.array(coords, dtype=float)
            return

        # 2) Cherche un unique sous-folder avec des placemarks
        subfolders = [c for c in folder_elem if strip_ns(c.tag) == "Folder"]
        if len(subfolders) == 1:
            sous = subfolders[0]
            placemarks_sous = [c for c in sous if strip_ns(c.tag) == "Placemark"]
            if placemarks_sous:
                tourn_name = None
                for child in folder_elem:
                    if strip_ns(child.tag) == "name" and child.text:
                        tourn_name = child.text.strip()
                        break
                if tourn_name:
                    coords = extrait_placemarks(placemarks_sous)
                    if coords:
                        route_points[tourn_name] = np.array(coords, dtype=float)
                return
            # Si ce sous-folder n'a pas de placemarks directs, on descend encore
            process_folder(sous)
            return

        # 3) Sinon, on descend dans chacun des sous-dossiers
        for subf in subfolders:
            process_folder(subf)

    # Déclenche la récursion sur tous les <Folder>
    for elem in root.iter():
        if strip_ns(elem.tag) == "Folder":
            process_folder(elem)

    return route_points


@st.cache_data(show_spinner=False)
def load_tournees_with_nn_thresholds(kml_filebuffer, default_threshold_km: float = 0.15):
    """
    Charge les tournées depuis un KML (non compressé) :
      - route_points_dict : { 'NomTournée': np.array([[lat,lon], ...]), ... }
      - thresholds_dict   : { 'NomTournée': seuil_km, ... }
      - hulls_dict        : { 'NomTournée': convex_hull (Shapely), ... }

    Seuil calculé comme médiane des distances NN × 2 (au minimum default_threshold_km).
    """
    # On charge d'abord le dict de points
    route_points_dict = load_points_from_kml(kml_filebuffer)
    if not route_points_dict:
        return {}, {}, {}

    # 1) On met en plat tous les points dans un DataFrame pour NN
    rows = []
    for tourn, pts in route_points_dict.items():
        for lat, lon in pts:
            rows.append({"Tournée": tourn, "Latitude": lat, "Longitude": lon})
    df_ref = pd.DataFrame(rows)

    # 2) On calcule les distances NN (k=2, car le plus proche voisin lui‐même a distance 0)
    nbrs = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(df_ref[["Latitude", "Longitude"]])
    distances, _ = nbrs.kneighbors(df_ref[["Latitude", "Longitude"]])
    # distances[:,0] = 0 (même point), distances[:,1] = distance au 2ᵉ voisin
    df_ref["NN_dist"] = distances[:, 1]

    thresholds = {}
    hulls = {}
    for tourn, grp in df_ref.groupby("Tournée"):
        # médiane de la NN distance
        median_nn = grp["NN_dist"].median()
        seuil = float(max(median_nn * 2, default_threshold_km))
        thresholds[tourn] = seuil

        # Construction du convex hull pour chaque tournée
        pts = route_points_dict[tourn]
        shp_pts = [Point(lon, lat) for lat, lon in pts]
        hull = MultiPoint(shp_pts).convex_hull
        hulls[tourn] = hull

    return route_points_dict, thresholds, hulls


@st.cache_data(show_spinner=False)
def geocode(address: str):
    """
    Géocode une adresse via Nominatim (OpenStreetMap) avec gestion du code 429 (backoff).
    On teste successivement :
      - l'adresse brute
      - l'adresse "nettoyée" (clean_address)
    Retourne (lat, lon) ou (None, None) si pas trouvé.
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
                # Petite pause et on abandonne cette variante
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
                # Pas de résultat -> on abandonne cette variante
                time.sleep(1)
                break

            elif resp.status_code == 429:
                # Trop de requêtes → backoff exponentiel
                time.sleep(backoff)
                backoff = min(backoff * 2, 5.0)
                continue

            else:
                time.sleep(1)
                break

    return None, None


def haversine_km(lat1, lon1, lat2, lon2):
    """
    Calcule la distance Haversine en km entre deux points (lat1,lon1) et (lat2,lon2).
    """
    R = 6371.0  # rayon terrestre en km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


# -------------------------------------------------------------------
#                                MAIN
# -------------------------------------------------------------------

def main():
    st.title("Attribution Automatique des Tournées PACA")
    st.write(
        "1) Uploadez votre fichier clients (Excel/CSV).\n"
        "2) Uploadez votre fichier de tournées au format KML (non compressé).\n"
        "3) L’app géocode et associe chaque client à la tournée la plus proche.\n"
        "4) Téléchargez le résultat enrichi au format Excel."
    )
    st.markdown("---")

    # ─── 1) Upload du fichier clients ─────────────────────────────────────────
    uploaded_clients = st.file_uploader(
        "🔹 1) Fichier client (Excel ou CSV)", type=["xlsx", "xls", "csv"], key="clients"
    )
    if not uploaded_clients:
        st.info("⏳ Vous devez d'abord uploader votre fichier de clients.")
        return

    # ─── 2) Upload du fichier KML (tournées) ────────────────────────────────
    uploaded_kml = st.file_uploader(
        "🔹 2) Fichier KML non compressé contenant les tournées", type=["kml"], key="kml"
    )
    if not uploaded_kml:
        st.info("⏳ Vous devez uploader votre KML de tournées.")
        return

    # 1a) Détection automatique de l’en-tête dans le fichier clients
    try:
        raw = pd.read_excel(uploaded_clients, header=None)
    except Exception:
        raw = pd.DataFrame()
    header_idx = 0
    for i, row in raw.iterrows():
        txt = " ".join(map(str, row.tolist())).lower()
        if any(k in txt for k in ("adresse", "cp", "codepostal", "ville")):
            header_idx = i
            break

    df_clients = pd.read_excel(uploaded_clients, header=header_idx)
    st.write("✅ Colonnes détectées dans le fichier client :", list(df_clients.columns))

    # 2) Détection automatique (ou manuelle) des colonnes Adresse, CP, Ville
    addr_cols = [c for c in df_clients.columns if any(w in c.lower() for w in ("adresse", "voie", "rue", "route", "chemin"))]
    cp_candidates = [c for c in df_clients.columns if "codepostal" in c.lower() or c.lower() == "cp"]
    ville_candidates = [c for c in df_clients.columns if "ville" in c.lower()]

    if not addr_cols:
        st.warning("Impossible de détecter automatiquement la colonne 'Adresse'. Choisissez-la manuellement ci‐dessous.")
        choix_addr = st.selectbox("Colonne Adresse ?", options=list(df_clients.columns), key="addr_choice")
        addr_cols = [choix_addr]

    if not cp_candidates:
        st.warning("Impossible de détecter automatiquement la colonne 'Code Postal'. Sélectionnez ou laissez vide.")
        choix_cp = st.selectbox("Colonne Code Postal ? (laisser vide si pas présent)", [None] + list(df_clients.columns), key="cp_choice")
        cp_col = choix_cp if choix_cp else None
    else:
        cp_col = cp_candidates[0]

    if not ville_candidates:
        st.warning("Impossible de détecter automatiquement la colonne 'Ville'. Sélectionnez ou laissez vide.")
        choix_ville = st.selectbox("Colonne Ville ? (laisser vide si pas présent)", [None] + list(df_clients.columns), key="ville_choice")
        ville_col = choix_ville if choix_ville else None
    else:
        ville_col = ville_candidates[0]

    st.write(f"→ Colonnes retenues : Adresse = {addr_cols}, Code Postal = {cp_col}, Ville = {ville_col}")
    st.markdown("---")

    # 3) Construction du champ adresse complète
    df_clients["_full_address"] = ""
    for c in addr_cols + ([cp_col] if cp_col else []) + ([ville_col] if ville_col else []):
        if c:
            df_clients["_full_address"] += df_clients[c].fillna("").astype(str) + " "
    df_clients["_full_address"] = df_clients["_full_address"].str.strip()

    # 4) Géocodage des adresses clients
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
        st.error("❌ Aucune adresse n'a pu être géocodée. Vérifiez vos colonnes Adresse/CP/Ville.")
        return
    st.success(f"✅ Géocodage terminé : {n_valid}/{total} adresses valides.")
    st.markdown("---")

    # 5) Extraction des tournées depuis le KML
    st.write("📂 Extraction des tournées depuis le KML…")
    # On peut donner directement l’UploadedFileIO à ET.parse (car il se comporte comme un fichier en mémoire)
    route_points_dict, thresholds_dict, hulls_dict = load_tournees_with_nn_thresholds(uploaded_kml)
    if not route_points_dict:
        st.error("❌ Le KML ne contient aucune tournée valide. Vérifiez la structure :")
        st.write("     – Chaque tourn\u00e9e doit être dans un <Folder> nommé.") 
        st.write("     – Chaque <Folder> (ou sous-<Folder>) doit contenir des <Placemark><Point><coordinates>.")
        return

    n_tournees = len(route_points_dict)
    st.success(f"🗂️ {n_tournees} tournées extraites depuis le KML.")
    st.markdown("---")

    # 6) Attribution des tournées pour chaque client
    st.write("🚚 Attribution des tournées (hull / NN fallback)…")
    progress_attr = st.progress(0)
    attribs = []

    for i, row in enumerate(df_clients.itertuples()):
        latc = getattr(row, "Latitude")
        lonc = getattr(row, "Longitude")
        if pd.isna(latc) or pd.isna(lonc):
            attribs.append("")  # pas de géocodage → on laisse vide
            progress_attr.progress((i + 1) / total)
            continue

        pt = Point(lonc, latc)
        choix = ""

        # 6.1) Si le point client est à l’intérieur du convex hull d’une tournée, on l’assigne immédiatement
        for tourn_name, hull in hulls_dict.items():
            if hull.contains(pt):
                choix = tourn_name
                break

        # 6.2) Sinon, fallback nearest‐neighbor (Haversine)
        if choix == "":
            best_tour = ""
            best_dist = float("inf")
            for tourn_name, pts in route_points_dict.items():
                arr = np.array(pts)  # shape = (N_points, 2) en (lat, lon)
                # On calcule la distance Haversine (en km) vers tous les points de la tournée
                # puis la distance minimale de ce cluster
                dists_km = np.vectorize(lambda la, lo: haversine_km(latc, lonc, la, lo))(arr[:,0], arr[:,1])
                dmin = float(dists_km.min())
                if dmin < best_dist:
                    best_dist = dmin
                    best_tour = tourn_name

            # On compare à son seuil calculé
            seuil = thresholds_dict.get(best_tour, default_threshold_km := 0.15)
            if best_dist <= seuil:
                choix = best_tour
            else:
                choix = ""

        attribs.append(choix)
        progress_attr.progress((i + 1) / total)

    df_clients["Tournée attribuée"] = attribs
    st.success("✅ Attribution des tournées terminée.")
    st.markdown("---")

    # 7) Export du tableau enrichi au format Excel
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_clients.to_excel(writer, index=False)
    st.download_button(
        "📥 Télécharger le fichier enrichi (.xlsx)",
        buffer.getvalue(),
        file_name="clients_tournees_enrichi.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    st.write("✅ Terminé.")

if __name__ == "__main__":
    main()
