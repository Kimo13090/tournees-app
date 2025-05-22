import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import difflib

# Chargement des bases
@st.cache_data
def load_data():
    base = pd.read_excel("Base_tournees_KML_coordonnees.xlsx")
    return base

# Fonction de recherche de la tournée
def retrouver_tournee(base, latitude, longitude):
    base["distance"] = base.apply(
        lambda row: geodesic((latitude, longitude), (row["latitude"], row["longitude"])).km,
        axis=1
    )
    resultat = base.loc[base["distance"].idxmin()]
    return resultat["Tournee"], resultat["distance"]

# Détection des colonnes similaires
def detect_columns(cols):
    targets = ["adresse", "code postal", "ville"]
    found = {}
    for target in targets:
        match = difflib.get_close_matches(target, [c.lower() for c in cols], n=1, cutoff=0.6)
        if match:
            original = [c for c in cols if c.lower() == match[0]][0]
            found[target] = original
    return found

# Interface Streamlit
st.title("🚚 Attribution automatique de tournées")

st.markdown("""
Téléversez un fichier Excel contenant les colonnes **adresse**, **code postal**, et **ville** (peu importe l'ordre ou les majuscules).
L'application tentera de retrouver automatiquement la tournée correspondante pour chaque ligne.
""")

uploaded_file = st.file_uploader("Téléversez un fichier Excel", type=["xlsx"])

if uploaded_file:
    df_input = pd.read_excel(uploaded_file)
    col_map = detect_columns(df_input.columns)

    if len(col_map) < 3:
        st.error("Le fichier doit contenir les colonnes : adresse, code postal, ville (ou noms similaires).")
    else:
        base = load_data()
        geolocator = Nominatim(user_agent="tournees-app")
        coordonnees = []

        for idx, row in df_input.iterrows():
            try:
                adresse_complete = f"{row[col_map['adresse']]}, {row[col_map['code postal']]}, {row[col_map['ville']]}"
                location = geolocator.geocode(adresse_complete)
                if location:
                    tournee, distance = retrouver_tournee(base, location.latitude, location.longitude)
                    coordonnees.append([location.latitude, location.longitude, tournee, distance])
                else:
                    coordonnees.append([None, None, "Non trouvée", None])
            except:
                coordonnees.append([None, None, "Erreur", None])

        df_input[["latitude", "longitude", "tournee", "distance_km"]] = pd.DataFrame(coordonnees)
        st.success("Tournées attribuées avec succès !")
        st.dataframe(df_input)

        st.download_button("📥 Télécharger le fichier avec tournées", data=df_input.to_excel(index=False), file_name="resultat_tournees.xlsx")
