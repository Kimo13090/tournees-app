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
        t_low = t.lower().strip(",.")
        # Abréviations
        if t_low in ("bd", "bld", "boul"):
            t = "boulevard"
        elif t_low in ("av", "av.", "aven"):
            t = "avenue"
        elif t_low in ("res", "res."):
            t = "résidence"
        # Supprime doublons consécutifs
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
    # Essais successifs
    for variant in (address, clean_address(address)):
        try:
            resp = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": variant, "format": "json", "limit": 1},
                headers=headers,
                timeout=5
            )
        except Exception:
            # En cas d'erreur réseau/timeout, on passe à la variante suivante
            continue

        if resp.status_code == 200 and resp.json():
            d = resp.json()[0]
            return float(d["lat"]), float(d["lon"])
        # Respect fair-use Nominatim
        time.sleep(1)
    # Échec
    return None, None

@st.cache_data
def load_tournees():
    df = pd.read_excel(TOURNEES_FILE)
    tourns = {}
    for name, group in df.groupby("Tournée"):
        lats = group["Latitude"].tolist()
        lons = group["Longitude"].tolist()
        # Centroïde
        centro_lat = sum(lats) / len(lats)
        centro_lon = sum(lons) / len(lons)
        # Distances internes
        dists = [
            distance_haversine(centro_lat, centro_lon, lat, lon)
            for lat, lon in zip(lats, lons)
        ]
        # Rayon au 90ème percentile
        rayon = pd.Series(dists).quantile(0.9)
        tourns[name] = {"centro": (centro_lat, centro_lon), "rayon": rayon}
    return tourns

def main():
    st.title("Attribution Automatique des Tournées PACA")
    st.write("Upload ton fichier clients (Adresse, CP, Ville…); l'app géocode et associe chaque client à sa tournée, ou marque HZ.")

    uploaded = st.file_uploader("Fichier Excel/CSV", type=["xlsx","xls","csv"])
    if not uploaded:
        return

    # Détection de l'en-tête
    raw = pd.read_excel(uploaded, header=None)
    header_idx = 0
    for i, row in raw.iterrows():
        combined = " ".join([str(x) for x in row.tolist()]).lower()
        if any(k in combined for k in ("adresse", "codepostal", "cp", "ville")):
            header_idx = i
            break
    df_clients = pd.read_excel(uploaded, header=header_idx)

    st.write("Colonnes détectées :", list(df_clients.columns))

    # Constitution du champ full_address
    addr_cols = [c for c in df_clients.columns if any(w in c.lower() for w in ("adresse","voie","rue","route","chemin"))]
    cp_col = next((c for c in df_clients.columns if "codepostal" in c.lower() or c.lower()=="cp"), "")
    ville_col = next((c for c in df_clients.columns if "ville" in c.lower()), "")

    df_clients["_full_address"] = ""
    for c in addr_cols + ([cp_col] if cp_col else []) + ([ville_col] if ville_col else []):
        df_clients["_full_address"] += df_clients[c].fillna("").astype(str) + " "

    # Géocodage
    lats, lons = [], []
    for addr in df_clients["_full_address"]:
        lat, lon = geocode(addr)
        lats.append(lat); lons.append(lon)
    df_clients["Latitude"] = lats
    df_clients["Longitude"] = lons

    # Attribution
    tourns = load_tournees()
    attribs, dists = [], []
    for _, row in df_clients.iterrows():
        latc, lonc = row["Latitude"], row["Longitude"]
        choix, dist_min = "HZ", None
        if pd.notna(latc) and pd.notna(lonc):
            # Recherche de la meilleure tournée
            for name, info in tourns.items():
                centro = info["centro"]
                rayon = info["rayon"]
                d = distance_haversine(latc, lonc, centro[0], centro[1])
                if d <= rayon and (dist_min is None or d < dist_min):
                    choix, dist_min = name, d
        attribs.append(choix)
        dists.append(round(dist_min, 2) if dist_min is not None else None)

    df_clients["Tournée attribuée"] = attribs
    df_clients["Distance (km)"] = dists

    # Export Excel
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_clients.to_excel(writer, index=False)
    st.download_button("Télécharger en .xlsx", buffer.getvalue(),
                       file_name="clients_tournees_enrichi.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__ == "__main__":
    main()
