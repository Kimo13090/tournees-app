import streamlit as st
import pandas as pd
import requests
import time
from math import radians, sin, cos, sqrt, atan2

# Config
USER_AGENT = "TourneeLocator/1.0 (ton_email@domaine.com)"

def distance_haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1); dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

@st.cache_data
def load_tournees():
    # Charge le fichier de base des tournées (xlsx ou csv) que tu devras placer à la racine
    df = pd.read_excel("Base_tournees_KML_coordonnees.xlsx")
    return df

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

def main():
    st.title("Attribution Automatique des Tournées")
    st.write("Téléverse ton fichier clients (Adresse, Code postal, Ville…). L'app géocode et attribue la tournée la plus proche.")
    
    uploaded = st.file_uploader("Fichier Excel/CSV", type=["xlsx", "xls", "csv"])
    if uploaded:
        if uploaded.name.endswith((".xlsx", ".xls")):
            df_clients = pd.read_excel(uploaded)
        else:
            df_clients = pd.read_csv(uploaded)
        
        # Concatène adresse
        df_clients["_full_address"] = df_clients["Adresse"].fillna("") + " " + \
                                      df_clients["Complément d'adresse"].fillna("") + " " + \
                                      df_clients["Code postal"].astype(str).fillna("") + " " + \
                                      df_clients["Ville"].fillna("")
        
        # Géocodage
        lats = []; lons = []
        for addr in df_clients["_full_address"]:
            lat, lon = geocode(addr)
            lats.append(lat); lons.append(lon)
            time.sleep(1)  # Respect fair-use
        df_clients["Latitude"] = lats; df_clients["Longitude"] = lons
        
        # Chargement des tournées
        df_tournees = load_tournees()
        
        # Calcul des distances et attribution
        assigned = []
        for _, client in df_clients.iterrows():
            best = (None, float("inf"))
            if pd.notna(client["Latitude"]) and pd.notna(client["Longitude"]):
                for _, tour in df_tournees.iterrows():
                    d = distance_haversine(client["Latitude"], client["Longitude"],
                                           tour["Latitude"], tour["Longitude"])
                    if d < best[1]:
                        best = (tour["Tournée"], d)
            assigned.append(best[0] or "Non trouvé")
        df_clients["Tournée attribuée"] = assigned
        
        st.dataframe(df_clients)
        st.download_button("Télécharger le fichier enrichi", df_clients.to_csv(index=False), "clients_tournees.csv")

if __name__ == "__main__":
    main()
