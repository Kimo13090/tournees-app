import streamlit as st
import pandas as pd
import numpy as np
import pgeocode
from sklearn.neighbors import BallTree
from rapidfuzz import process, fuzz

# --- Helpers ----------------------------------------------------------

@st.cache_data
def load_base(path="Base_tournees_KML_coordonnees.xlsx"):
    df = pd.read_excel(path, engine="openpyxl")
    # On standardise le nom des colonnes
    df = df.rename(columns=str.strip)
    # Vérifie qu'on a bien latitude, longitude, Tournee
    assert {"latitude","longitude","Tournee"}.issubset(df.columns), \
        "La base doit contenir au moins les colonnes latitude, longitude et Tournee"
    # converts deg → rad
    coords = np.deg2rad(df[["latitude","longitude"]].values)
    tree = BallTree(coords, metric="haversine")
    return df, tree

def detect_columns(df):
    choices = df.columns.tolist()
    def pick(colname):
        best, score, _ = process.extractOne(colname, choices, scorer=fuzz.partial_ratio)
        return best if score>75 else None
    return {
        "adresse": pick("adresse"),
        "code postal": pick("code postal"),
        "ville":       pick("ville"),
    }

def get_postal_centroid(cp_series):
    nomi = pgeocode.Nominatim("fr")
    # renvoie un array Nx2 radian
    lat = nomi.query_postal_code(cp_series.astype(str)).latitude.fillna(0).values
    lon = nomi.query_postal_code(cp_series.astype(str)).longitude.fillna(0).values
    return np.deg2rad(np.vstack([lat, lon]).T)

def assign_tournees(df_input, base_df, tree, k=1):
    # on ne tient QUE du centrioïde du code postal
    pts = get_postal_centroid(df_input["code postal"])
    dist, ind = tree.query(pts, k=k)
    df_input["Tournee trouvée"] = base_df.iloc[ind[:,0]]["Tournee"].values
    df_input["Distance (km)"]   = (dist[:,0] * 6371).round(2)
    return df_input

# --- Streamlit UI ------------------------------------------------------

st.set_page_config(page_title="Attribution automatique de tournées", layout="wide")
st.title("🚚 Attribution automatique de tournées")
st.markdown("""
Téléversez un fichier Excel contenant au minimum **une colonne d'adresse**, **de code postal** et **de ville**.
L'application va estimer (à partir du centroid postal) la tournée la plus proche.
""")

uploaded = st.file_uploader("Téléversez un Excel (.xlsx)", type="xlsx")
if uploaded:
    df = pd.read_excel(uploaded, engine="openpyxl")
    cols = detect_columns(df)
    if None in cols.values():
        st.error("Impossible de détecter automatiquement les colonnes. Vérifiez que vous avez bien `adresse`, `code postal` et `ville`.")
        st.stop()

    df = df.rename(columns={v:k for k,v in cols.items()})
    base_df, tree = load_base()
    result = assign_tournees(df, base_df, tree)
    st.success("✅ Attribution terminée !")
    st.dataframe(result)

    # bouton d'export
    towrite = pd.ExcelWriter("résultat.xlsx", engine="openpyxl")
    result.to_excel(towrite, index=False)
    towrite.save()
    st.download_button("⬇️ Télécharger le résultat", "résultat.xlsx", "résultat.xlsx")
