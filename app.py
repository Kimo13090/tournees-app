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
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def clean_address(addr: str) -> str:
    """Nettoie l'adresse en supprimant doublons, abréviations et accents."""
    s = unidecode(addr)
    tokens = s.split()
    cleaned, prev = [], None
    for t in tokens:
        tl = t.lower().strip(".,")
        if tl in ("bd","bld","boul"):
            t = "boulevard"
        elif tl in ("av","av.","aven"):
            t = "avenue"
        elif tl in ("res","res."):
            t = "residence"
        if t.lower() != prev:
            cleaned.append(t)
            prev = t.lower()
    return " ".join(cleaned)

@st.cache_data(show_spinner=False)
def geocode(address: str):
    """
    Géocode une adresse via Nominatim :
    1) essai sur l'adresse brute
    2) essai sur l'adresse nettoyée
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
        except:
            continue
        if resp.status_code == 200 and resp.json():
            d = resp.json()[0]
            return float(d["lat"]), float(d["lon"])
        time.sleep(1)  # Respect du fair-use Nominatim
    return None, None

@st.cache_data
def load_tournees_with_thresholds():
    """
    Lit la base des tournées (avec Latitude, Longitude, Tournée) et
    calcule pour chaque tournée :
      - son centroïde GPS (moyenne des lat/lon)
      - le 90e percentile des distances de chacun de ses points historiques au centroïde
    Renvoie :
      - df_ref : DataFrame brute (Latitude, Longitude, Tournée)
      - dict_centroides : {nom_tournée: (lat_cent, lon_cent)}
      - dict_seuils     : {nom_tournée: seuil_90_percentile_km}
    """
    df_ref = pd.read_excel(TOURNEES_FILE)
    dict_centroides = {}
    dict_seuils = {}

    for name, grp in df_ref.groupby("Tournée"):
        # Liste des lat/lon historiques
        lats = grp["Latitude"].tolist()
        lons = grp["Longitude"].tolist()
        # Calcul du centroïde
        centro_lat = sum(lats) / len(lats)
        centro_lon = sum(lons) / len(lons)
        dict_centroides[name] = (centro_lat, centro_lon)

        # Calcul des distances historiques au centroïde
        dists = [
            distance_haversine(centro_lat, centro_lon, lat, lon)
            for lat, lon in zip(lats, lons)
        ]
        # Seuil = 90e percentile
        seuil = pd.Series(dists).quantile(0.90)
        dict_seuils[name] = seuil

    return df_ref, dict_centroides, dict_seuils

def main():
    st.title("Attribution Automatique des Tournées PACA")
    st.write("1) Uploade ton fichier clients (Adresse, CP, Ville…)\n"
             "2) Laisse le système déterminer automatiquement la tournée la plus appropriée\n"
             "3) Télécharge le résultat (sans seuil manuel)")

    uploaded = st.file_uploader("Fichier Excel/CSV", type=["xlsx","xls","csv"])
    if not uploaded:
        return

    # --- 1) Détection automatique de l'en-tête ---
    raw = pd.read_excel(uploaded, header=None)
    header_idx = 0
    for i, row in raw.iterrows():
        txt = " ".join(map(str, row.tolist())).lower()
        if any(k in txt for k in ("adresse", "cp", "codepostal", "ville")):
            header_idx = i
            break

    df = pd.read_excel(uploaded, header=header_idx)
    st.write("Colonnes détectées :", list(df.columns))

    # --- 2) Construction du champ d'adresse complète ---
    addr_cols = [
        c for c in df.columns 
        if any(w in c.lower() for w in ("adresse", "voie", "rue", "route", "chemin"))
    ]
    cp_col = next((c for c in df.columns if "codepostal" in c.lower() or c.lower() == "cp"), None)
    ville_col = next((c for c in df.columns if "ville" in c.lower()), None)

    df["_full_address"] = ""
    for c in addr_cols + ([cp_col] if cp_col else []) + ([ville_col] if ville_col else []):
        df["_full_address"] += df[c].fillna("").astype(str) + " "

    # --- 3) Géocodage avec barre de progression ---
    total = len(df)
    st.write(f"🔍 Géocodage de {total} adresses…")
    progress = st.progress(0)
    lats, lons = [], []
    for i, addr in enumerate(df["_full_address"]):
        lat, lon = geocode(addr)
        lats.append(lat)
        lons.append(lon)
        progress.progress((i + 1) / total)
    df["Latitude"] = lats
    df["Longitude"] = lons
    st.success("✅ Géocodage terminé")

    # --- 4) Chargement des tournées historique + seuils dynamiques ---
    df_ref, dict_centroides, dict_seuils = load_tournees_with_thresholds()

    # --- 5) Attribution automatique sans intervention manuelle ---
    st.write("🚚 Attribution des tournées… (pas de seuil manuel)")
    progress = st.progress(0)
    attribs = []
    for i, row in enumerate(df.itertuples()):
        latc, lonc = getattr(row, "Latitude"), getattr(row, "Longitude")
        if pd.isna(latc) or pd.isna(lonc):
            attribs.append("")  # client non géocodé → on laisse vide
        else:
            # Calcul des distances du client à chaque centroïde de tournée
            dists_cent = {
                name: distance_haversine(latc, lonc, centro[0], centro[1])
                for name, centro in dict_centroides.items()
            }
            # Tournée la plus proche
            min_name = min(dists_cent, key=dists_cent.get)
            min_dist = dists_cent[min_name]
            # Si distance <= seuil 90e percentile de cette tournée → on attribue
            if min_dist <= dict_seuils[min_name]:
                attribs.append(min_name)
            else:
                attribs.append("")  # trop éloigné, on laisse vide
        progress.progress((i + 1) / total)

    df["Tournée attribuée"] = attribs
    st.success("✅ Attribution terminée")

    # --- 6) Export en .xlsx ---
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    st.download_button(
        "Télécharger le fichier enrichi (.xlsx)",
        buffer.getvalue(),
        file_name="clients_tournees.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if __name__ == "__main__":
    main()

