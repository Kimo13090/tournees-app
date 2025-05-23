import streamlit as st
import pandas as pd
import requests
import time
from math import radians, sin, cos, sqrt, atan2

# --- Configuration ---
USER_AGENT = "TourneeLocator/1.0 (ton_email@domaine.com)"  # Remplace par ton email de contact

# --- Fonctions utilitaires ---
def distance_haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Rayon de la Terre en km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

@st.cache_data
def load_tournees():
    # Charge le fichier de base des tournées (xlsx ou csv) dans la racine
    try:
        return pd.read_excel("Base_tournees_KML_coordonnees.xlsx")
    except FileNotFoundError:
        st.error("Le fichier Base_tournees_KML_coordonnees.xlsx est introuvable dans le répertoire de l'app.")
        st.stop()

@st.cache_data
def geocode(address):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": address, "format": "json", "limit": 1}
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, params=params, headers=headers)
    if resp.status_code == 200 and resp.json():
        data = resp.json()[0]
        return float(data["lat"]), float(data["lon"])
    return None, None

# --- Application principale ---
def main():
    st.title("Attribution Automatique des Tournées")
    st.write("Téléversez votre fichier clients (Excel/CSV) contenant l'adresse, le code postal, la ville... L'app géocode et attribue la tournée la plus proche.")

    # Téléversement du fichier
    uploaded = st.file_uploader("Fichier Excel/CSV", type=["xlsx", "xls", "csv"])
    if not uploaded:
        return

    # Lecture du DataFrame
    if uploaded.name.endswith((".xlsx", ".xls")):
        df_clients = pd.read_excel(uploaded)
    else:
        df_clients = pd.read_csv(uploaded)

    # Affichage des colonnes pour debug
    st.write("**Colonnes détectées :**", list(df_clients.columns))

    # --- Construction de l'adresse complète ---
    # Normalisation des noms de colonnes existantes
    cols_map = {col.lower().strip().replace(" ", ""): col for col in df_clients.columns}
    # Recherche automatique des champs
    adresse_col    = cols_map.get("adresse") or cols_map.get("adresseclient") or cols_map.get("adresseprincipale") or ""
    complement_col = cols_map.get("complementdadresse") or cols_map.get("complémentdadresse") or ""
    cp_col         = cols_map.get("codepostal") or cols_map.get("cp") or ""
    ville_col      = cols_map.get("ville") or cols_map.get("commune") or ""

    # Assemblage sans planter si absent
    parts = []
    for key in (adresse_col, complement_col, cp_col, ville_col):
        if key and key in df_clients.columns:
            parts.append(df_clients[key].fillna("").astype(str))
        else:
            parts.append(pd.Series([""] * len(df_clients)))
    df_clients["_full_address"] = parts[0] + " " + parts[1] + " " + parts[2] + " " + parts[3]

    # Géocodage
    lats, lons = [], []
    for addr in df_clients["_full_address"]:
        lat, lon = geocode(addr)
        lats.append(lat)
        lons.append(lon)
        time.sleep(1)  # Respect fair-use Nominatim
    df_clients["Latitude"] = lats
    df_clients["Longitude"] = lons

    # Chargement et attribution des tournées
    df_tournees = load_tournees()
    assigned = []
    for _, client in df_clients.iterrows():
        best = (None, float("inf"))
        # Filtre par zone si possible
        zone = client.get("Ville") or client.get("ville") or client.get(cp_col)
        if zone:
            subset = df_tournees[df_tournees["Zone"].astype(str).str.lower() == str(zone).lower()]
        else:
            subset = df_tournees
        # Calcul des distances
        lat_c, lon_c = client["Latitude"], client["Longitude"]
        if pd.notna(lat_c) and pd.notna(lon_c):
            for _, tour in subset.iterrows():
                d = distance_haversine(lat_c, lon_c, tour["Latitude"], tour["Longitude"])
                if d < best[1]:
                    best = (tour["Tournée"], d)
        assigned.append(best[0] or "Non trouvé")
    df_clients["Tournée attribuée"] = assigned

    # Affichage et téléchargement
    st.dataframe(df_clients)
    st.download_button(
        "Télécharger le fichier enrichi", df_clients.to_csv(index=False).encode('utf-8'), "clients_tournees.csv"
    )

if __name__ == "__main__":
    main()
