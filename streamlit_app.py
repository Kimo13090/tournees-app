import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from difflib import get_close_matches

# --- Chargement de la base tournée (lire votre fichier de tournées + coordonnées) ---
@st.cache_data
def load_base():
    # Assurez-vous que ce fichier se trouve à la racine du dépôt
    return pd.read_excel("Base_tournees_KML_coordonnees.xlsx")

# --- Détection fuzzy des colonnes utilisateur ---
def detect_columns(df):
    lower2orig = {c.lower(): c for c in df.columns}
    mapping = {}
    for target in ["adresse", "code postal", "ville"]:
        m = get_close_matches(target, lower2orig.keys(), n=1, cutoff=0.6)
        if m:
            mapping[target] = lower2orig[m[0]]
    return mapping

# --- Recherche de la tournée la plus proche via géodésie ---
def retrieve_tournee(base_df, lat, lon):
    # on suppose que base_df contient les colonnes "Tournee", "Lat", "Lon"
    distances = base_df.apply(
        lambda r: geodesic((lat, lon), (r["Lat"], r["Lon"])).meters, axis=1
    )
    idx = distances.idxmin()
    return base_df.loc[idx, "Tournee"]

def main():
    st.set_page_config(page_title="Attribution tournées", layout="wide")
    st.title("🚚 Attribution automatique de tournées")
    st.markdown("""
Téléversez un fichier Excel contenant les colonnes **adresse**, **code postal**, et **ville**.  
L'application tentera de retrouver automatiquement la tournée correspondante pour chaque ligne.
""")

    uploaded_file = st.file_uploader("Téléversez un fichier Excel", type=["xlsx"])
    if not uploaded_file:
        return

    # Lecture du fichier utilisateur
    df_input = pd.read_excel(uploaded_file)
    col_map = detect_columns(df_input)
    if len(col_map) < 3:
        st.error("Le fichier doit contenir les colonnes : adresse, code postal, ville (ou équivalents).")
        return

    # Chargement de la base tournée
    base = load_base()
    geolocator = Nominatim(user_agent="tournees-app", timeout=10)

    # Traitement ligne à ligne
    tours = []
    for _, row in df_input.iterrows():
        query = (
            f"{row[col_map['adresse']]}, "
            f"{row[col_map['code postal']]}, "
            f"{row[col_map['ville']]}"
        )
        loc = geolocator.geocode(query)
        if loc:
            tour = retrieve_tournee(base, loc.latitude, loc.longitude)
        else:
            tour = "Non trouvé"
        tours.append(tour)

    # Ajout et téléchargement
    df_input["Tournée"] = tours
    towrite = pd.ExcelWriter("resultats.xlsx", engine="openpyxl")
    df_input.to_excel(towrite, index=False)
    towrite.close()

    st.success("Attribution terminée ! Téléchargez le fichier ci-dessous.")
    with open("resultats.xlsx", "rb") as f:
        st.download_button(
            label="📥 Télécharger les résultats",
            data=f,
            file_name="resultats.xlsx"
        )

if __name__ == "__main__":
    main()
