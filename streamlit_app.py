import streamlit as st
import pandas as pd
from geopy.distance import geodesic

# Chargement des bases
@st.cache_data
def load_data():
    base = pd.read_excel("Base_tournees_KML_coordonnees.xlsx")
    return base

# Fonction de recherche de la tournée
def retrouver_tournee(base, latitude, longitude):
    base["distance"] = base.apply(
        lambda row: geodesic((latitude, longitude), (row["latitude"], row["longitude"])).meters,
        axis=1
    )
    resultat = base.loc[base["distance"].idxmin()]
    return resultat["Tournee"], resultat["distance"]

# Interface Streamlit
st.set_page_config(page_title="Détection de tournée", page_icon="🚚")
st.title("🚚 Attribution automatique de tournées")

st.markdown("""
Téléversez un fichier Excel contenant les colonnes **adresse**, **code postal**, et **ville**.

L'application tentera de retrouver automatiquement la tournée correspondante pour chaque ligne.
""")

uploaded_file = st.file_uploader("Téléversez un fichier Excel", type=["xlsx"])

if uploaded_file is not None:
    df_clients = pd.read_excel(uploaded_file)

    if all(col in df_clients.columns for col in ["adresse", "code postal", "ville"]):
        st.success("Fichier correctement chargé !")

        # Géocodage via Google Maps API ou Nominatim (prétraitement attendu)
        if "latitude" not in df_clients.columns or "longitude" not in df_clients.columns:
            st.error("Les colonnes 'latitude' et 'longitude' sont manquantes. Géocodez-les avant.")
        else:
            base = load_data()
            tournees = []
            distances = []

            for _, row in df_clients.iterrows():
                tournee, distance = retrouver_tournee(base, row["latitude"], row["longitude"])
                tournees.append(tournee)
                distances.append(distance)

            df_clients["Tournee_deduite"] = tournees
            df_clients["Distance_m"] = distances

            st.dataframe(df_clients)

            st.download_button(
                label="🔹 Télécharger le résultat",
                data=df_clients.to_excel(index=False),
                file_name="clients_avec_tournees.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.error("Le fichier doit contenir les colonnes : adresse, code postal, ville")
