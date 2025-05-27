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

def clean_address(addr: str) -> str:
    """Nettoie l'adresse en supprimant doublons, abréviations et accents."""
    s = unidecode(addr)
    tokens = s.split()
    cleaned, prev = [], None
    for t in tokens:
        tl = t.lower().strip(".,")
        if tl in ("bd","bld","boul"): t="boulevard"
        elif tl in ("av","av.","aven"): t="avenue"
        elif tl in ("res","res."): t="residence"
        if t.lower() != prev:
            cleaned.append(t)
            prev = t.lower()
    return " ".join(cleaned)

@st.cache_data(show_spinner=False)
def geocode(address: str):
    """Géocode via Nominatim: adresse brute puis nettoyée."""
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

# --- Application principale ---
def main():
    st.title("Attribution Automatique des Tournées PACA")
    st.write("1) Uploade ton fichier clients (Adresse, CP, Ville…)\n2) Ajuste la distance max\n3) Télécharge ton fichier enrichi")

    uploaded = st.file_uploader("Fichier Excel/CSV", type=["xlsx","xls","csv"])
    if not uploaded:
        return

    # 1) Détection de l'en-tête
    raw = pd.read_excel(uploaded, header=None)
    header_idx = 0
    for i, row in raw.iterrows():
        txt = " ".join(map(str, row.tolist())).lower()
        if any(k in txt for k in ("adresse","cp","codepostal","ville")):
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

    # 3) Géocodage avec progression
    lats, lons = [], []
    total = len(df)
    progress_bar = st.progress(0)
    st.write(f"🔍 Géocodage de {total} adresses…")
    for i, addr in enumerate(df["_full_address"]):
        lat, lon = geocode(addr)
        lats.append(lat); lons.append(lon)
        progress_bar.progress((i+1)/total)
    df["Latitude"] = lats
    df["Longitude"] = lons
    st.success("✅ Géocodage terminé")

    # 4) Lecture de la base historique des tournées
    df_ref = pd.read_excel(TOURNEES_FILE)

    # 5) Slider de distance max (km)
    max_dist = st.slider(
        "Distance max pour attribuer une tournée (km)",
        0.1, 10.0, 1.0, step=0.1
    )

    # 6) Attribution par plus proche voisin
    st.write("🚚 Attribution des tournées…")
    attribs = []
    for idx, row in df.iterrows():
        latc, lonc = row["Latitude"], row["Longitude"]
        if pd.isna(latc) or pd.isna(lonc):
            attribs.append("HZ")
            continue

        # Calcul des distances à tous les points historiques
        dists = df_ref.apply(
            lambda r: distance_haversine(latc, lonc, r["Latitude"], r["Longitude"]),
            axis=1
        )
        i_min = dists.idxmin()
        d_min = dists.iat[i_min]

        if d_min <= max_dist:
            attribs.append(df_ref.at[i_min, "Tournée"])
        else:
            attribs.append("HZ")

        progress_bar.progress((idx+1)/total)

    df["Tournée attribuée"] = attribs
    st.success("✅ Attribution terminée")

    # 7) Export Excel
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
