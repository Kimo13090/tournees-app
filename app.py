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

    Structure attendue du KML : 
       <kml>
         <Document>
           <Folder>                ← Dépositaire (ex. "MARSEILLE")
             <name>MARSEILLE</name>
             <Folder>              ← Tournée (ex. "A001 - ALLAUCH 1")
               <name>A001 - …</name>
               …Sous‐folders et Placemark…
             </Folder>
             <Folder>…Tournée suivante…</Folder>
           </Folder>
           <Folder>…Autre Dépositaire…</Folder>
         </Document>
       </kml>

    On parcourt chaque <Folder> de 2ᵉ niveau (sous “Dépositaire”),
    on collecte **tous** les <Placemark> descendants, on récupère les coordonnées
    (lon,lat) et on les stocke sous forme de np.array([[lat,lon],[lat,lon],…]).
    """
    tree = ET.parse(kml_path)
    root = tree.getroot()

    # Cherche l'élément <Document> si présent
    doc_elem = None
    for child in root:
        if strip_ns(child.tag) == "Document":
            doc_elem = child
            break
    if doc_elem is None:
        doc_elem = root  # pas de <Document>, on travaille depuis la racine

    route_points = {}

    def extract_coords_from_placemark(pm_elem):
        """
        Lit un <Placemark> et renvoie (lat, lon) ou None si échec.
        On cherche la première balise <coordinates> contenant "lon,lat,alt".
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

    # Parcours de chaque “dépositaire” (= Folder enfant de <Document>)
    for depos_elem in doc_elem:
        if strip_ns(depos_elem.tag) != "Folder":
            continue

        # Parcours de chaque “tournée” (= Folder enfant du dépositaire)
        for tourn_elem in depos_elem:
            if strip_ns(tourn_elem.tag) != "Folder":
                continue

            # Récupérer le <name> du folder “tournée”
            tourn_name = None
            for kid in tourn_elem:
                if strip_ns(kid.tag) == "name" and kid.text:
                    tourn_name = kid.text.strip()
                    break
            if not tourn_name:
                continue

            # Collecter **tous** les Placemark descendants
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
    Charge les tournées depuis un KML (non compressé) :
      - route_points_dict : { 'NomTournée': np.array([[lat,lon], ...]), ... }
      - thresholds_dict   : { 'NomTournée': seuil_km, ... }
      - polygons_dict     : { 'NomTournée': shapely Polygon (ou MultiPolygon), ... }

    Pour chaque tournée :
      1) On extrait son nuage de points (lat, lon).
      2) Pour chaque point, on calcule la distance Haversine au point le plus proche
         (médiane des nearest neighbors). On en tire un “seuil” = max(2×médiane, default).
      3) Pour créer un “corridor” (buffer) autour de la tournée, on met un buffer
         d’un rayon = (seuil km en degrés ≃ seuil/111) autour de **chaque point**,
         puis on fait l’union de tous ces cercles pour obtenir un Polygon.
    Retourne (route_points_dict, thresholds_dict, polygons_dict).
    """
    route_points_dict = load_points_from_kml(kml_file)
    if not route_points_dict:
        return None, None, None

    # 1) Construire un DataFrame plat pour calcul NN
    rows = []
    for tourn_name, pts in route_points_dict.items():
        for lat, lon in pts:
            rows.append({"Tournée": tourn_name, "Latitude": lat, "Longitude": lon})
    df_ref = pd.DataFrame(rows)

    # 2) Calculer la nearest‐neighbor (Haversine) pour chaque point / même tourn
    def haversine_array(lat1, lon1, lat2_arr, lon2_arr):
        lat1r = radians(lat1)
        lon1r = radians(lon1)
        lat2r = np.radians(lat2_arr)
        lon2r = np.radians(lon2_arr)
        dlat = lat2r - lat1r
        dlon = lon2r - lon1r
        a = np.sin(dlat/2)**2 + np.cos(lat1r)*np.cos(lat2r)*np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return 6371.0 * c

    nn_distances = []
    for tourn_name, grp in df_ref.groupby("Tournée"):
        latitudes = grp["Latitude"].values
        longitudes = grp["Longitude"].values
        n_pts = len(latitudes)
        if n_pts <= 1:
            # Si seul point, pas de voisin → distance infinite (on ne buffer pas trop petit)
            nn_distances.extend([np.inf] * n_pts)
            continue

        # Pour chaque point, calculer la distance la plus petite vers les autres
        for i in range(n_pts):
            lat_i = latitudes[i]
            lon_i = longitudes[i]
            dists = haversine_array(lat_i, lon_i, latitudes, longitudes)
            dists[i] = np.inf
            nn_distances.append(dists.min())

    df_ref["NN_dist_km"] = nn_distances

    # 3) Calculer seuils et générer des “buffers union”
    thresholds = {}
    polygons = {}
    for tourn_name, grp in df_ref.groupby("Tournée"):
        median_nn = grp["NN_dist_km"].replace(np.inf, 0).median()
        seuil = float(max(median_nn * 2, default_threshold_km))
        thresholds[tourn_name] = seuil

        # Rayon en degrés approximatif
        pts = route_points_dict[tourn_name]
        radius_deg = seuil / 111.0  # ≃ 1° = 111 km

        # On bufferise chaque point en “cercle” de rayon radius_deg puis union
        circles = []
        for (lat, lon) in pts:
            circle = Point(lon, lat).buffer(radius_deg)
            circles.append(circle)
        union_poly = unary_union(circles)
        polygons[tourn_name] = union_poly

    return route_points_dict, thresholds, polygons


@st.cache_data(show_spinner=False)
def geocode(address: str):
    """
    Géocode une adresse via Nominatim (OpenStreetMap), avec backoff si statut 429.
    On teste l’adresse brute, puis l’adresse nettoyée (clean_address).
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
                time.sleep(backoff)
                backoff = min(backoff * 2, 5.0)
                continue

            else:
                time.sleep(1)
                break

    return None, None


def haversine_km(lat1, lon1, lat2, lon2):
    """
    Distance Haversine (en km) entre (lat1,lon1) et (lat2,lon2).
    """
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


# ---------------------------------------------------------------------------
#                               FONCTION PRINCIPALE
# ---------------------------------------------------------------------------

def main():
    st.title("Attribution Automatique des Tournées PACA")
    st.write("""
    1) Uploadez votre fichier clients (Excel/CSV)  
    2) L’app géocode chaque client et l’associe à la tournée la plus proche  
    3) Téléchargez le résultat enrichi (.xlsx)  
    """)

    # --- 1) Uploader le fichier client (Excel ou CSV) ---
    uploaded = st.file_uploader("Fichier client (Excel/CSV)", type=["xlsx", "xls", "csv"])
    if not uploaded:
        return

    # 2) Lecture **brute** sans header pour que toutes les colonnes apparaissent comme “Unnamed: 0,1,2…”
    df_raw = pd.read_excel(uploaded, header=None)
    st.write("Aperçu des 5 premières lignes (lecture sans en-tête) :")
    st.dataframe(df_raw.head(5), use_container_width=True)

    # 3) Montrer les colonnes détectées (indices 0,1,2…) pour que l’utilisateur choisisse
    cols_df = pd.DataFrame({"Index": df_raw.columns})
    st.write("Liste des colonnes détectées (indices) :")
    st.dataframe(cols_df, use_container_width=True)

    # 4) Sélection manuelle : Adresse, Code Postal, Ville
    st.warning("Impossible de détecter automatiquement la colonne 'Adresse' (pas d'en-tête). " +
               "Choisissez manuellement parmi les indices ci-dessous.")
    choix_addr = st.selectbox("Colonne Adresse (index)", options=list(df_raw.columns.map(str)))
    addr_col = int(choix_addr)

    st.warning("Impossible de détecter automatiquement la colonne 'Code Postal'. " +
               "Sélectionnez-la ou laissez vide.")
    liste_cp = ["None"] + list(df_raw.columns.map(str))
    choix_cp = st.selectbox("Colonne Code Postal (index)", options=liste_cp, index=0)
    cp_col = None if choix_cp == "None" else int(choix_cp)

    st.warning("Impossible de détecter automatiquement la colonne 'Ville'. " +
               "Sélectionnez-la ou laissez vide.")
    liste_ville = ["None"] + list(df_raw.columns.map(str))
    choix_ville = st.selectbox("Colonne Ville (index)", options=liste_ville, index=0)
    ville_col = None if choix_ville == "None" else int(choix_ville)

    st.write(f"→ Colonnes utilisées : Adresse=Index {addr_col}, CP={cp_col}, Ville={ville_col}")

    # 5) Recharger **idéalement** le DataFrame en prenant la même ligne comme header,
    #    mais ici on n'a PAS d'entête du tout, donc on convertit directement le df_raw :
    df_clients = df_raw.copy()
    # On va accéder aux colonnes via leur indice numérique (0,1,2…it.)

    # 6) Construire la colonne “_full_address”
    def safe_str(x):
        return "" if pd.isna(x) else str(x)

    full_addresses = []
    for idx, row in df_clients.iterrows():
        parts = []
        parts.append(safe_str(row[addr_col]))
        if cp_col is not None:
            parts.append(safe_str(row[cp_col]))
        if ville_col is not None:
            parts.append(safe_str(row[ville_col]))
        full_addresses.append(" ".join([p for p in parts if p.strip()] ))
    df_clients["_full_address"] = full_addresses

    # 7) Géocodage de la colonne _full_address
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

    # 8) Charger les tournées depuis le KML
    st.write("📂 Extraction des tournées depuis le fichier KML…")
    KML_TOURNEES_FILE = "abonnes_portes_analyste_tournee.kml"
    route_points_dict, thresholds_dict, polygons_dict = load_tournees_with_nn_thresholds(KML_TOURNEES_FILE)
    if route_points_dict is None:
        st.error("❌ Le KML ne contient aucune tournée valide. Vérifiez la structure (Folder → Placemark).")
        return
    n_tournees = len(route_points_dict)
    st.success(f"🗂 {n_tournees} tournées extraites depuis le KML.")

    # 9) Attribution des tournées aux clients
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

        # 9.1) Si le point se trouve DANS le “corridor” (polygon) d’une tournée → on affecte
        for tourn_name, poly in polygons_dict.items():
            if poly.contains(pt):
                choix = tourn_name
                break

        # 9.2) Sinon, fallback nearest‐neighbor
        if choix == "":
            best_tour = ""
            best_dist = float("inf")
            for tourn_name, pts in route_points_dict.items():
                arr = np.array(pts)  # (lat, lon)
                dists = np.vectorize(lambda la, lo: haversine_km(latc, lonc, la, lo))(
                    arr[:,0], arr[:,1]
                )
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

    # 10) Export du fichier enrichi en .xlsx
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
