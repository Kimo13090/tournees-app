import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import unicodedata

# Détection automatique des colonnes utiles
def detect_columns_auto(df):
    columns = {col: normalize(col) for col in df.columns}
    adresse_col = next((col for col, norm in columns.items() if "adresse" in norm), None)
    postal_col = next((col for col, norm in columns.items() if "codepostal" in norm or "cp" == norm), None)
    ville_col = next((col for col, norm in columns.items() if "ville" in norm), None)
    return adresse_col, postal_col, ville_col

def normalize(text):
    text = str(text).lower()
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    return text.replace(" ", "").replace("_", "")

@st.cache_data
def load_data():
    return pd.read_excel("Base_tournees_KML_coordonnees.xlsx")

def retrouver_tournee(base, latitude, longitude):
    base["distance"] = base.apply(
        lambda row: geodesic((latitude, longitude), (row["Latitude"], row["Longitude"])).meters,
        axis=1
    )
    result = base.loc[base["distance"].idxmin()]
    return result["Tournee"], result["distance"]

# Interface utilisateur
st.title("🚛 Attribution automatique de tournées")
st.markdown("""
Téléversez un fichier Excel contenant des adresses.
L'application retrouvera automatiquement la tournée correspondante.
""")

uploaded_file = st.file_uploader("Téléversez un fichier Excel", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    adresse_col, postal_col, ville_col = detect_columns_auto(df)

    if not all([adresse_col, postal_col, ville_col]):
        st.error("Colonnes adresse/code postal/ville non détectées automatiquement. Merci de vérifier votre fichier.")
    else:
        base_coords = load_data()
        geolocator = Nominatim(user_agent="tournees-app")

        resultats = []
        for idx, row in df.iterrows():
            adresse_complete = f"{row[adresse_col]}, {row[postal_col]} {row[ville_col]}"
            try:
                location = geolocator.geocode(adresse_complete)
                if location:
                    tournee, distance = retrouver_tournee(base_coords, location.latitude, location.longitude)
                else:
                    tournee, distance = "Non trouvé", None
            except:
                tournee, distance = "Erreur", None
            resultats.append((adresse_complete, tournee, distance))

        resultat_df = pd.DataFrame(resultats, columns=["Adresse complète", "Tournée attribuée", "Distance (m)"])
        st.success("Traitement terminé.")
        st.dataframe(resultat_df)
        st.download_button("🔹 Télécharger le résultat", resultat_df.to_csv(index=False).encode('utf-8'), file_name="resultat_tournees.csv", mime="text/csv")
