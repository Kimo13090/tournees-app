import streamlit as st
import pandas as pd
from geopy.distance import geodesic

# Chargement des bases
@st.cache_data
def load_data():
    base = pd.read_excel("Base_tournees_KML_coordonnees.xlsx")
    return base

# Détection intelligente des colonnes
@st.cache_data
def detect_columns(df):
    col_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if "adresse" in col_lower and "e-mail" not in col_lower:
            col_map["adresse"] = col
        elif "code postal" in col_lower:
            col_map["code postal"] = col
        elif "ville" in col_lower:
            col_map["ville"] = col
    return col_map

# Fonction de recherche de la tournée
def retrouver_tournee(base, latitude, longitude):
    base["distance"] = base.apply(
        lambda row: geodesic((latitude, longitude), (row["Latitude"], row["Longitude"])).meters,
        axis=1
    )
    resultat = base.loc[base["distance"].idxmin()]
    return resultat["Tournee"], resultat["distance"]

# Interface Streamlit
st.title("🚚 Attribution automatique de tournées")

st.markdown("""
Téléversez un fichier Excel contenant les colonnes **adresse**, **code postal**, et **ville**.
L'application tentera de retrouver automatiquement la tournée correspondante pour chaque ligne.
""")

uploaded_file = st.file_uploader("Téléversez un fichier Excel", type=["xlsx"])

if uploaded_file:
    df_input = pd.read_excel(uploaded_file)
    col_map = detect_columns(df_input)

    if len(col_map) < 3:
        st.error("Le fichier doit contenir les colonnes : adresse, code postal, ville (ou noms similaires).")
    else:
        base = load_data()

        from geopy.geocoders import Nominatim
        geolocator = Nominatim(user_agent="tournees-app")

        coordonnees = []
        for idx, row in df_input.iterrows():
            adresse_complete = f"{row[col_map['adresse']]}, {row[col_map['code postal']]}, {row[col_map['ville']]}"
            try:
                location = geolocator.geocode(adresse_complete)
                if location:
                    tournee, distance = retrouver_tournee(base, location.latitude, location.longitude)
                else:
                    tournee, distance = "Non trouvé", None
            except:
                tournee, distance = "Erreur", None
            coordonnees.append((tournee, distance))

        df_input["Tournee"] = [t for t, _ in coordonnees]
        df_input["Distance (m)"] = [d if d else "" for _, d in coordonnees]

        st.success("🎉 Attribution terminée !")
        st.dataframe(df_input)

        # Téléchargement du résultat
        from io import BytesIO
        output = BytesIO()
        df_input.to_excel(output, index=False)
        st.download_button("🔗 Télécharger le fichier résultat", output.getvalue(), file_name="resultat_tournees.xlsx")
