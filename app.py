import streamlit as st
import pandas as pd
import requests
import time
import io
from math import radians, sin, cos, sqrt, atan2
from shapely.geometry import MultiPoint, Point

# --- Configuration ---
USER_AGENT = "TourneeLocator/1.0 (contact@votredomaine.com)"
TOURNEES_FILE = "Base_tournees_KML_coordonnees.xlsx"

# --- Fonctions utilitaires ---
def distance_haversine(lat1, lon1, lat2, lon2):
    """Calcule la distance en km entre deux points GPS."""
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def clean_address(addr: str) -> str:
    """Nettoie l'adresse en supprimant doublons et abréviations."""
    tokens = addr.split()
    cleaned = []
    prev = None
    for t in tokens:
        tl = t.lower().strip('.,')
        # Remplacements courants
        if tl in ("bd", "bld", "boul"): t = "boulevard"
        elif tl in ("av", "av.", "aven"): t = "avenue"
        elif tl in ("res", "res."): t = "résidence"
        # Supprime doublon consécutif
        if t.lower() != prev:
            cleaned.append(t)
            prev = t.lower()
    return " ".join(cleaned)

@st.cache_data(show_spinner=False)
def geocode(address: str):
    """
    Géocode une adresse via Nominatim :
      1) variante brute
      2) variante nettoyée
    """
    headers = {"User-Agent": USER_AGENT}
    for variant in (address, clean_address(address)):
        try:
            resp = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": variant, "format": "json", "limit": 1},
                headers=headers,
                timeout=5
            )
        except Exception:
            continue
        if resp.status_code == 200 and resp.json():
            d = resp.json()[0]
            return float(d["lat"]), float(d["lon"])
        # Respect du fair-use
        time.sleep(1)
    return None, None

@st.cache_data
def load_tournees():
    """Charge les tournées et construit leur convex hull."""
    df = pd.read_excel(TOURNEES_FILE)
    hulls = {}
    for name, grp in df.groupby("Tournée"):
        points = [Point(lon, lat) for lat, lon in zip(grp["Latitude"], grp["Longitude"])]
        hulls[name] = MultiPoint(points).convex_hull
    return hulls

# --- Application principale ---
def main():
    st.title("Attribution Automatique des Tournées PACA")
    st.write("Upload ton fichier clients (Adresse, CP, Ville…), l'app géocode et associe chaque client.")

    uploaded = st.file_uploader("Fichier Excel/CSV", type=["xlsx","xls","csv"])
    if not uploaded:
        return

    # 1) Détection de l'en-tête
    raw = pd.read_excel(uploaded, header=None)
    header_idx = 0
    for i, row in raw.iterrows():
        txt = " ".join(map(str, row.tolist())).lower()
        if any(k in txt for k in ("adresse", "cp", "codepostal", "ville")):
            header_idx = i
            break
    df_clients = pd.read_excel(uploaded, header=header_idx)
    st.write("Colonnes détectées :", list(df_clients.columns))

    # 2) Construction du champ d'adresse complète
    addr_cols = [c for c in df_clients.columns if any(w in c.lower() for w in ("adresse","voie","rue","route","chemin"))]
    cp_col = next((c for c in df_clients.columns if "codepostal" in c.lower() or c.lower()=="cp"), None)
    ville_col = next((c for c in df_clients.columns if "ville" in c.lower()), None)
    df_clients["_full_address"] = ""
    for c in addr_cols + ([cp_col] if cp_col else []) + ([ville_col] if ville_col else []):
        df_clients["_full_address"] += df_clients[c].fillna("").astype(str) + " "

    # 3) Géocodage avec barre de progression
    lats, lons = [], []
    total = len(df_clients)
    progress_bar = st.progress(0)
    st.write(f"🔍 Géocodage de {total} adresses…")
    for i, addr in enumerate(df_clients["_full_address"]):
        lat, lon = geocode(addr)
        lats.append(lat)
        lons.append(lon)
        progress_bar.progress((i + 1) / total)
    df_clients["Latitude"] = lats
    df_clients["Longitude"] = lons
    st.success("✅ Géocodage terminé")

    # 4) Attribution avec convex hull + fallback distance
    tourns = load_tournees()
    # Préparer centroids pour fallback
    centroids = {name: (poly.centroid.y, poly.centroid.x) for name, poly in tourns.items()}
    FALLBACK_RADIUS_KM = 1.0  # seuil pour fallback
    attribs = []
    for _, row in df_clients.iterrows():
        choix = "HZ"
        latc, lonc = row["Latitude"], row["Longitude"]
        if pd.notna(latc) and pd.notna(lonc):
            pt = Point(lonc, latc)
            # test inclusion
            for name, poly in tourns.items():
                if poly.intersects(pt):
                    choix = name
                    break
            else:
                # fallback: nearest centroid
                best_name, best_dist = None, float('inf')
                for name, (cy, cx) in centroids.items():
                    d = distance_haversine(latc, lonc, cy, cx)
                    if d < best_dist:
                        best_name, best_dist = name, d
                if best_dist <= FALLBACK_RADIUS_KM:
                    choix = best_name
        attribs.append(choix)
    # debug lengths
    st.write(f"Nb lignes clients : {len(df_clients)}, Nb attribs : {len(attribs)}")
    df_clients["Tournée attribuée"] = attribs
    st.success("✅ Attribution terminée")

    # 5) Export Excel
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_clients.to_excel(writer, index=False)
    st.download_button(
        "Télécharger le fichier enrichi (.xlsx)",
        buffer.getvalue(),
        file_name="clients_tournees_enrichi.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        st.error("🚨 Une erreur est survenue lors de l'exécution de l'application :")
        st.error(str(e))
        st.text(traceback.format_exc())
