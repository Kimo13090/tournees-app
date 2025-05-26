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
    R = 6371
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
        if tl in ("bd", "bld", "boul"): t = "boulevard"
        elif tl in ("av", "av.", "aven"): t = "avenue"
        elif tl in ("res", "res."): t = "résidence"
        if t.lower() != prev:
            cleaned.append(t)
            prev = t.lower()
    return " ".join(cleaned)

@st.cache_data(show_spinner=False)
def geocode(address: str):
    """
    Géocode une adresse via Nominatim:
      1) adresse brute
      2) adresse nettoyée
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
        time.sleep(1)
    return None, None

@st.cache_data
def load_tournees():
    """Charge les tournées et construit leurs polygones convex hull."""
    df = pd.read_excel(TOURNEES_FILE)
    tourns = {}
    for name, grp in df.groupby("Tournée"):
        pts = [Point(lon, lat) for lat, lon in zip(grp["Latitude"], grp["Longitude"])]
        hull = MultiPoint(pts).convex_hull
        tourns[name] = hull
    return tourns

# --- Application Streamlit ---
def main():
    st.title("Attribution Automatique des Tournées PACA")
    st.write("Upload ton fichier clients (Adresse, CP, Ville…), l'app géocode et associe chaque client à sa tournée, ou marque HZ.")

    uploaded = st.file_uploader("Fichier Excel/CSV", type=["xlsx","xls","csv"])
    if not uploaded:
        return

    # Détection de l'en-tête
    raw = pd.read_excel(uploaded, header=None)
    header_idx = 0
    for i, row in raw.iterrows():
        text = " ".join(str(x) for x in row.tolist()).lower()
        if any(k in text for k in ("adresse", "codepostal", "cp", "ville")):
            header_idx = i
            break
    df_clients = pd.read_excel(uploaded, header=header_idx)

    st.write("Colonnes détectées :", list(df_clients.columns))

    # Concaténation des champs adresse
    addr_cols = [c for c in df_clients.columns if any(w in c.lower() for w in ("adresse","voie","rue","route","chemin"))]
    cp_col = next((c for c in df_clients.columns if "codepostal" in c.lower() or c.lower()=="cp"), None)
    ville_col = next((c for c in df_clients.columns if "ville" in c.lower()), None)
    df_clients["_full_address"] = ""
    for c in addr_cols + ([cp_col] if cp_col else []) + ([ville_col] if ville_col else []):
        df_clients["_full_address"] += df_clients[c].fillna("").astype(str) + " "

    # --- GÉOCODAGE ---
# Initialisation de la barre de progression
lats, lons = [], []
total = len(df)
progress_bar = st.progress(0)
st.write(f"🔍 Géocodage de {total} adresses...")
for i, addr in enumerate(df["_full_address"]):
    lat, lon = geocode(addr)
    lats.append(lat)
    lons.append(lon)
    # Mise à jour de la barre
    progress_bar.progress((i + 1) / total)
# Assignation des résultats
df["Latitude"] = lats
df["Longitude"] = lons
st.success(f"✅ Géocodage terminé : {len([x for x in lats if x is not None])}/{total} adresses géocodées")

# --- ATTRIBUTION VIA CONVEX HULL ---
# Debug : vérifier longueurs avant attribution
st.write(f"Nb de lignes clients = {len(df)}, valeurs prochaines = {len(df)}")

hulls = load_tournees()
attribs = []
for idx, row in df.iterrows():
    latc, lonc = row["Latitude"], row["Longitude"]
    choix = "HZ"
    if pd.notna(latc) and pd.notna(lonc):
        pt = Point(lonc, latc)
        for name, hull in hulls.items():
            if hull.contains(pt):
                choix = name
                break
    attribs.append(choix)

# Assignation finale
if len(attribs) == len(df):
    df["Tournée attribuée"] = attribs
else:
    st.error("❌ Erreur interne : mismatch entre le nombre de clients et d'attributions !")

    tourns = load_tournees()
    attribs = []
    with st.spinner("🔄 Attribution des tournées..."):
        for idx, row in df_clients.iterrows():
            latc, lonc = row["Latitude"], row["Longitude"]
            pt = Point(lonc, latc) if pd.notna(latc) and pd.notna(lonc) else None
            choix = "HZ"
            if pt:
                for name, hull in tourns.items():
                    if hull.contains(pt):
                        choix = name
                        break
            attribs.append(choix)
    st.success("✅ Attribution terminée")

    # Debug longueur
    st.write(f"Lignes clients: {len(df_clients)}, attributs: {len(attribs)}")

    # Assignation
    df_clients["Tournée attribuée"] = attribs

    # Export Excel
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_clients.to_excel(writer, index=False)
    st.download_button("📥 Télécharger en .xlsx", buffer.getvalue(),
                       file_name="clients_tournees_enrichi.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__ == "__main__":
    main()
