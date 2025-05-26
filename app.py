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

def distance_haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    c = 2*atan2(sqrt(a), sqrt(1-a))
    return R * c

def clean_address(addr: str) -> str:
    tokens = addr.split()
    cleaned, prev = [], None
    for t in tokens:
        tl = t.lower().strip(".,")
        if tl in ("bd","bld","boul"): t="boulevard"
        elif tl in ("av","av.","aven"): t="avenue"
        elif tl in ("res","res."): t="résidence"
        if t.lower()!=prev:
            cleaned.append(t); prev=t.lower()
    return " ".join(cleaned)

@st.cache_data(show_spinner=False)
def geocode(address: str):
    headers={"User-Agent":USER_AGENT}
    for variant in (address, clean_address(address)):
        try:
            resp=requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q":variant,"format":"json","limit":1},
                headers=headers,timeout=5
            )
        except:
            continue
        if resp.status_code==200 and resp.json():
            d=resp.json()[0]
            return float(d["lat"]), float(d["lon"])
        time.sleep(1)
    return None,None

@st.cache_data
def load_tournees():
    df=pd.read_excel(TOURNEES_FILE)
    tourns={}
    for name,grp in df.groupby("Tournée"):
        pts=[Point(lon,lat) for lat,lon in zip(grp["Latitude"], grp["Longitude"])]
        hull=MultiPoint(pts).convex_hull
        tourns[name]=hull
    return tourns

def main():
    st.title("Attribution Automatique des Tournées PACA")
    st.write("Upload ton fichier clients (Adresse, CP, Ville...).")

    uploaded=st.file_uploader("Fichier Excel/CSV", type=["xlsx","xls","csv"])
    if not uploaded:
        return

    # Détecter en-tête
    raw=pd.read_excel(uploaded,header=None)
    header_idx=0
    for i,row in raw.iterrows():
        txt=" ".join(map(str,row.tolist())).lower()
        if any(k in txt for k in("adresse","codepostal","cp","ville")):
            header_idx=i
            break
    df=pd.read_excel(uploaded,header=header_idx)
    st.write("Colonnes détectées :",list(df.columns))

    # Concat adresses
    addr_cols=[c for c in df.columns if any(w in c.lower() for w in("adresse","voie","rue","route","chemin"))]
    cp_col=next((c for c in df.columns if "codepostal" in c.lower() or c.lower()=="cp"),None)
    ville_col=next((c for c in df.columns if "ville" in c.lower()),None)
    df["_full_address"]=""
    for c in addr_cols+([cp_col] if cp_col else[])+([ville_col] if ville_col else[]):
        df["_full_address"]+=df[c].fillna("").astype(str)+" "

    # Géocodage
    lats,lons=[],[]
    with st.spinner("Géocodage en cours..."):
        for i,addr in enumerate(df["_full_address"]):
            lat,lon=geocode(addr)
            lats.append(lat); lons.append(lon)
            st.progress((i+1)/len(df))
    df["Latitude"]=lats; df["Longitude"]=lons
    st.success("Géocodage terminé")

    # Attribution
    hulls=load_tournees()
    attribs=[]
    # Debug lengths
    st.write(f"Nb lignes clients : {len(df)}, attributs initial : debugging")
    for idx,row in df.iterrows():
        latc,lonc=row["Latitude"],row["Longitude"]
        choix="HZ"
        if pd.notna(latc) and pd.notna(lonc):
            pt=Point(lonc,latc)
            for name,hull in hulls.items():
                if hull.contains(pt):
                    choix=name
                    break
        attribs.append(choix)
    # Vérif longueur
    st.write(f"Nb valeurs attribs : {len(attribs)}")
    df["Tournée attribuée"]=attribs
    st.success("Attribution terminée")

    # Export
    buf=io.BytesIO()
    with pd.ExcelWriter(buf,engine="openpyxl") as w:
        df.to_excel(w,index=False)
    st.download_button("Télécharger .xlsx",buf.getvalue(),
                       file_name="clients_tournees.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__=="__main__":
    main()
