import streamlit as st
import pandas as pd
import requests
import time
import io
from math import radians, sin, cos, sqrt, atan2
from unidecode import unidecode

# --- Configuration ---
USER_AGENT = "TourneeLocator/1.0 (contact@votredomaine.com)"
TOURNEES_FILE = "Base_tournees_KML_coordonnees.xlsx"

# --- Fonctions utilitaires ---
def distance_haversine(lat1, lon1, lat2, lon2):
    """Calcule la distance en km entre deux points GPS."""
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

@st.cache_data(show_spinner=False)
def geocode(address: str):
    """Géocode une adresse via Nominatim (brute puis nettoyée)."""
    headers = {"User-Agent": USER_AGENT}
    cleaned = unidecode(address)
    for variant in (cleaned, cleaned):  # tenter deux fois identique pour garantir
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
def load_base_tournees():
    """Charge la base des tournées avec leurs points GPS."""
    df = pd.read_excel(TOURNEES_FILE)
    groups = {}
    for name, grp in df.groupby("Tournée"):
        coords = list(zip(grp["Latitude"], grp["Longitude"]))
        groups[name] = coords
    return groups

# --- Application ---
def main():
    st.title("Attribution Tournées PACA (Proche + Nearest)")
    st.write("Upload un fichier client (Adresse, CP, Ville…). L’app attribue la tournée la plus proche si à l’intérieur du seuil, sinon HZ.")

    uploaded = st.file_uploader("Fichier Excel/CSV", type=["xlsx","xls","csv"])
    if not uploaded:
        return

    # Lecture avec détection d'en-tête
    raw = pd.read_excel(uploaded, header=None)
    header_idx = 0
    for i, row in raw.iterrows():
        txt = " ".join(map(str, row.tolist())).lower()
        if any(k in txt for k in ("adresse", "cp", "codepostal", "ville")):
            header_idx = i
            break
    df = pd.read_excel(uploaded, header=header_idx)
    st.write("Colonnes détectées :", list(df.columns))

    # Concaténation et normalisation de l'adresse
    addr_cols = [c for c in df.columns if any(w in c.lower() for w in ("adresse","voie","rue","route","chemin"))]
    cp_col = next((c for c in df.columns if "codepostal" in c.lower() or c.lower()=="cp"), None)
    ville_col = next((c for c in df.columns if "ville" in c.lower()), None)
    df["_full_address"] = ""
    for c in addr_cols + ([cp_col] if cp_col else []) + ([ville_col] if ville_col else []):
        df["_full_address"] += df[c].fillna("").astype(str) + " "
    df["_full_address"] = df["_full_address"].apply(lambda x: unidecode(x).strip())

    # Géocodage
    total = len(df)
    st.write(f"🔍 Géocodage de {total} adresses…")
    progress_bar = st.progress(0)
    lats, lons = [], []
    for i, addr in enumerate(df["_full_address"]):
        lat, lon = geocode(addr)
        lats.append(lat); lons.append(lon)
        progress_bar.progress((i+1)/total)
    df["Latitude"] = lats; df["Longitude"] = lons
    st.success("✅ Géocodage terminé")

    # Chargement base tournées
    base = load_base_tournees()

    # Seuil de proximité (km)
    seuil = st.number_input("Seuil de proximité (km) pour attribution, sinon HZ", min_value=0.1, value=0.5, step=0.1)

    # Attribution par distance au plus proche point de chaque tournée
    st.write("🔄 Attribution des tournées…")
    attribs = []
    for _, row in df.iterrows():
        latc, lonc = row["Latitude"], row["Longitude"]
        if pd.isna(latc) or pd.isna(lonc):
            attribs.append("HZ")
            continue
        # calcule min distance par tournée
        best = (None, float('inf'))
        for name, coords in base.items():
            for lat2, lon2 in coords:
                d = distance_haversine(latc, lonc, lat2, lon2)
                if d < best[1]: best = (name, d)
        # assignation selon seuil
        choix = best[0] if best[1] <= seuil else "HZ"
        attribs.append(choix)
    df["Tournée attribuée"] = attribs
    st.success("✅ Attribution terminée")

    # Export Excel
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    st.download_button(
        "Télécharger (.xlsx)", buf.getvalue(),
        file_name="clients_tournees_enrichi.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if __name__ == "__main__":
    main()
