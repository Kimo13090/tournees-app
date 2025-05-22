import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from difflib import get_close_matches

# --- Charge et détecte les colonnes de la base tournée ---
@st.cache_data
def load_base():
    df = pd.read_excel("Base_tournees_KML_coordonnees.xlsx")
    # détection fuzzy pour « tournée », « latitude », « longitude »
    lower = {c.lower(): c for c in df.columns}
    col_tournee = lower[get_close_matches("tournee", lower, n=1, cutoff=0.6)[0]]
    col_lat = lower[get_close_matches("lat", lower, n=1, cutoff=0.6)[0]]
    col_lon = lower[get_close_matches("lon", lower, n=1, cutoff=0.6)[0]]
    return df, col_tournee, col_lat, col_lon

# --- Détecte fuzzy les colonnes du fichier client ---
def detect_input_columns(df):
    lower = {c.lower(): c for c in df.columns}
    mapping = {}
    for target in ("adresse", "code postal", "ville"):
        m = get_close_matches(target, lower, n=1, cutoff=0.6)
        if m:
            mapping[target] = lower[m[0]]
    return mapping

# --- Recherche de tournée la plus proche ---
def retrieve_tournee(base_df, col_tour, col_lat, col_lon, lat, lon):
    # calcule la distance à chaque point de la base
    dists = base_df.apply(
        lambda r: geodesic((lat, lon), (r[col_lat], r[col_lon])).meters,
        axis=1)
    idx = dists.idxmin()
    return base_df.loc[idx, col_tour]

def main():
    st.set_page_config(page_title="Attribution tournées", layout="wide")
    st.title("🚚 Attribution automatique de tournées")

    st.markdown("""
Téléversez un fichier Excel contenant les colonnes **adresse**,  
**code postal** et **ville** (ou équivalents).  
L'application cherchera la tournée la plus proche pour chaque ligne.
""")

    uploaded = st.file_uploader("Votre fichier client", type=["xlsx"])
    if not uploaded:
        return

    # lit le fichier client
    df_in = pd.read_excel(uploaded)
    col_map = detect_input_columns(df_in)
    if len(col_map) < 3:
        st.error("Votre fichier doit contenir adresse, code postal et ville.")
        return

    # charge la base tournée et ses noms de colonnes
    base_df, col_tour, col_lat, col_lon = load_base()

    # instancie le géocodeur
    geoloc = Nominatim(user_agent="tournees-app", timeout=10)

    # boucle de traitement
    tours = []
    for _, row in df_in.iterrows():
        q = (
            f"{row[col_map['adresse']]}, "
            f"{row[col_map['code postal']]}, "
            f"{row[col_map['ville']]}"
        )
        loc = geoloc.geocode(q)
        if loc:
            tour = retrieve_tournee(base_df, col_tour, col_lat, col_lon,
                                     loc.latitude, loc.longitude)
        else:
            tour = "Non trouvé"
        tours.append(tour)

    # ajoute la colonne et propose le téléchargement
    df_in["Tournée"] = tours
    with pd.ExcelWriter("resultats.xlsx", engine="openpyxl") as w:
        df_in.to_excel(w, index=False)

    st.success("Terminé ! Téléchargez vos résultats ci-dessous.")
    with open("resultats.xlsx", "rb") as f:
        st.download_button("📥 Télécharger", f, "resultats.xlsx")

if __name__ == "__main__":
    main()
