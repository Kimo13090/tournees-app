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
# Tampon (en degrés) ~ 50 m à l'équateur
BUFFER_DEGREES = 0.0005  

def clean_address(addr: str) -> str:
    tokens = addr.split()
    cleaned, prev = [], None
    for t in tokens:
        tl = t.lower().strip(".,")
        if tl in ("bd","bld","boul"): t="boulevard"
        elif tl in ("av","av.","aven"): t="avenue"
        elif tl in ("res","res."): t="résidence"
        if t.lower() != prev:
            cleaned.append(t)
            prev = t.lower()
    return " ".join(cleaned)

@st.cache_data(show_spinner=False)
def geocode(address: str):
    headers = {"User-Agent": USER_AGENT}
    for variant in (address, clean_address(address)):
        try:
            resp = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": variant, "format": "json", "limit": 1},
                headers=headers,
                timeout=5
            )
        except:
            continue
        if resp.status_code == 200 and resp.json():
            d = resp.json()[0]
            return float(d["lat"]), float(d["lon"])
        time.sleep(1)
    return None, None

@st.cache_data
def load_tournees():
    """
    Charge les tournées depuis Excel et construit pour chacune 
    un polygone convex hull légèrement tamponné pour inclure ses frontières.
    """
    df = pd.read_excel(TOURNEES_FILE)
    tourns = {}
    for name, grp in df.groupby("Tournée"):
        pts = [Point(lon, lat) for lat, lon in zip(grp["Latitude"], grp["Longitude"])]
        hull = MultiPoint(pts).convex_hull
        # on ajoute un petit tampon pour capturer les points sur le bord
        tourns[name] = hull.buffer(BUFFER_DEGREES)
    return tourns

def main():
    st.title("Attribution Automatique des Tournées PACA")
    st.write("Upload ton fichier clients (Adresse, CP, Ville…), l'app géocode et associe chaque client.")

    uploaded = st.file_uploader("Fichier Excel/CSV", type=["xlsx","xls","csv"])
    if not uploaded:
        return

    # 1) Détection de la ligne d'en-tête
    raw = pd.read_excel(uploaded, header=None)
    header_idx = 0
    for i, row in raw.iterrows():
        txt = " ".join(map(str, row.tolist())).lower()
        if any(k in txt for k in ("adresse", "cp", "codepostal", "ville")):
            header_idx = i
            break
    df = pd.read_excel(uploaded, header=header_idx)
    st.write("Colonnes détectées :", list(df.columns))

    # 2) Construction du champ d'adresse complète
    addr_cols = [c for c in df.columns if any(w in c.lower() for w in ("adresse","voie","rue","route","chemin"))]
    cp_col = next((c for c in df.columns if "codepostal" in c.lower() or c.lower()=="cp"), None)
    ville_col = next((c for c in df.columns if "ville" in c.lower()), None)
    df["_full_address"] = ""
    for c in addr_cols + ([cp_col] if cp_col else []) + ([ville_col] if ville_col else []):
        df["_full_address"] += df[c].fillna("").astype(str) + " "

    # 3) Géocodage avec barre de progression
    lats, lons = [], []
    total = len(df)
    progress_bar = st.progress(0)
    st.write(f"🔍 Géocodage de {total} adresses…")
    for i, addr in enumerate(df["_full_address"]):
        lat, lon = geocode(addr)
        lats.append(lat); lons.append(lon)
        progress_bar.progress((i + 1) / total)
    df["Latitude"] = lats
    df["Longitude"] = lons
    st.success("✅ Géocodage terminé")

    # 4) Attribution avec convex hull tamponné
    tourns = load_tournees()
    attribs = []
    for _, row in df.iterrows():
        choix = "HZ"
        latc, lonc = row["Latitude"], row["Longitude"]
        if pd.notna(latc) and pd.notna(lonc):
            pt = Point(lonc, latc)
            for name, poly in tourns.items():
                # on utilise intersects pour capturer bord et intérieur
                if poly.intersects(pt):
                    choix = name
                    break
        attribs.append(choix)
    df["Tournée attribuée"] = attribs
    st.success("✅ Attribution terminée")

    # 5) Export Excel
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    st.download_button(
        "Télécharger le fichier enrichi (.xlsx)",
        buffer.getvalue(),
        file_name="clients_tournees_enrichi.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if __name__ == "__main__":
    main()
