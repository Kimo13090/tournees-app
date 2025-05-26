import streamlit as st
import pandas as pd
import requests
import time
import io
from math import radians, sin, cos, sqrt, atan2

# --- Configuration ---
USER_AGENT = "TourneeLocator/1.0 (contact@votredomaine.com)"
TOURNEES_FILE = "Base_tournees_KML_coordonnees.xlsx"

def distance_haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def clean_address(addr: str) -> str:
    # Supprime tokens doublon et remplace abréviations courantes
    tokens = addr.split()
    cleaned = []
    prev = None
    for t in tokens:
        t_low = t.lower()
        # abréviations
        if t_low in ("bd", "boul", "bld"): t = "boulevard"
        elif t_low in ("av", "av.", "aven"): t = "avenue"
        elif t_low in ("res", "res."): t = "résidence"
        # supprimer doublons consécutifs
        if t.lower() != prev:
            cleaned.append(t)
        prev = t.lower()
    return " ".join(cleaned)

def geocode(address: str):
    for variant in (address, clean_address(address)):
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": variant, "format": "json", "limit": 1}
        headers = {"User-Agent": USER_AGENT}
        resp = requests.get(url, params=params, headers=headers)
        if resp.status_code == 200 and resp.json():
            d = resp.json()[0]
            return float(d["lat"]), float(d["lon"])
        time.sleep(1)
    return None, None

@st.cache_data
def load_tournees():
    df = pd.read_excel(TOURNEES_FILE)
    # Calcul centre et rayon 90e percentile
    tourns = {}
    for name, group in df.groupby("Tournée"):
        lats = group["Latitude"].tolist()
        lons = group["Longitude"].tolist()
        centro_lat = sum(lats)/len(lats)
        centro_lon = sum(lons)/len(lons)
        # distances
        dists = [distance_haversine(centro_lat, centro_lon, lat, lon)
                 for lat, lon in zip(lats, lons)]
        rayon = pd.Series(dists).quantile(0.9)
        tourns[name] = {"centro": (centro_lat, centro_lon), "rayon": rayon}
    return tourns

def main():
    st.title("Attribution Automatique des Tournées PACA")
    st.write("Upload ton fichier clients (Adresse, CP, Ville…); l'app géocode et associe chaque client à sa tournée, ou marque HZ si hors zone.")

    uploaded = st.file_uploader("Fichier Excel/CSV", type=["xlsx","xls","csv"])
    if not uploaded:
        return

    # 1) Lecture et détection de l'en-tête
    raw = pd.read_excel(uploaded, header=None)
    header_idx = 0
    for i, row in raw.iterrows():
        combined = " ".join([str(x) for x in row.tolist()]).lower()
        if any(k in combined for k in ("adresse", "cp", "code postal", "ville")):
            header_idx = i
            break
    df_clients = pd.read_excel(uploaded, header=header_idx)

    st.write("Colonnes détectées :", list(df_clients.columns))

    # 2) Concaténation de adresses
    # On recherche automatiquement les colonnes
    cols = [c.lower().replace(" ", "") for c in df_clients.columns]
    addr_cols = [c for c in df_clients.columns if "adresse" in c.lower() or "voie" in c.lower() or "rue" in c.lower() or "route" in c.lower()]
    cp_col = next((c for c in df_clients.columns if "codepostal" in c.lower() or "cp"==c.lower()), "")
    ville_col = next((c for c in df_clients.columns if "ville" in c.lower()), "")

    df_clients["_full_address"] = ""
    for c in addr_cols + ([cp_col] if cp_col else []) + ([ville_col] if ville_col else []):
        df_clients["_full_address"] += df_clients[c].fillna("").astype(str) + " "

    # 3) Géocodage
    lats, lons = [], []
    for addr in df_clients["_full_address"]:
        lat, lon = geocode(addr)
        lats.append(lat); lons.append(lon)
    df_clients["Latitude"] = lats
    df_clients["Longitude"] = lons

    # 4) Attribution robuste
    tourns = load_tournees()
    attribs, dists = [], []
    for _, row in df_clients.iterrows():
        latc, lonc = row["Latitude"], row["Longitude"]
        best_name, best_dist = None, float("inf")
        for name, info in tourns.items():
            centro = info["centro"]; rayon = info["rayon"]
            if pd.notna(latc) and pd.notna(lonc):
                d = distance_haversine(latc, lonc, centro[0], centro[1])
                if d <= rayon and d < best_dist:
                    best_name, best_dist = name, d
        attribs.append(best_name or "HZ")
        dists.append(round(best_dist, 2) if best_name else None)
    df_clients["Tournée attribuée"] = attribs
    df_clients["Distance (km)"] = dists

    # 5) Télécharger en Excel
    to_export = df_clients
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        to_export.to_excel(writer, index=False)
    st.download_button("Télécharger le fichier enrichi (.xlsx)", buffer.getvalue(),
                       file_name="clients_tournees_enrichi.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__ == "__main__":
    main()
