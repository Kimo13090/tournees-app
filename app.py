from unidecode import unidecode

FALLBACK_RADIUS_KM = 1.0  # Seuil de proximité pour fallback

def main():
    st.title("Attribution Automatique des Tournées PACA")
    st.write("Upload ton fichier clients (Adresse, CP, Ville…), l'app géocode et associe chaque client.")

    uploaded = st.file_uploader("Fichier Excel/CSV", type=["xlsx","xls","csv"])
    if not uploaded:
        return

    # 1) Détection de l'en-tête
    raw = pd.read_excel(uploaded, header=None)
    header_idx = 0
    for i, row in raw.iterrows():
        txt = " ".join(map(str, row.tolist())).lower()
        if any(k in txt for k in ("adresse", "cp", "codepostal", "ville")):
            header_idx = i
            break
    df_clients = pd.read_excel(uploaded, header=header_idx)
    st.write("Colonnes détectées :", list(df_clients.columns))

    # 2) Construction du champ d'adresse complète (normalisation unidecode)
    addr_cols = [c for c in df_clients.columns if any(w in c.lower() for w in ("adresse","voie","rue","route","chemin"))]
    cp_col = next((c for c in df_clients.columns if "codepostal" in c.lower() or c.lower()=="cp"), None)
    ville_col = next((c for c in df_clients.columns if "ville" in c.lower()), None)
    df_clients["_full_address"] = ""
    for c in addr_cols + ([cp_col] if cp_col else []) + ([ville_col] if ville_col else []):
        df_clients["_full_address"] += df_clients[c].fillna("").astype(str) + " "
    df_clients["_full_address"] = df_clients["_full_address"].apply(lambda x: unidecode(x))

    # 3) Géocodage
    lats, lons = [], []
    total = len(df_clients)
    progress_bar = st.progress(0)
    st.write(f"🔍 Géocodage de {total} adresses…")
    for i, addr in enumerate(df_clients["_full_address"]):
        lat, lon = geocode(addr)
        lats.append(lat); lons.append(lon)
        progress_bar.progress((i + 1) / total)
    df_clients["Latitude"] = lats
    df_clients["Longitude"] = lons
    st.success("✅ Géocodage terminé")

    # 4) Préparer centroids pour fallback
    hulls = load_tournees()  # dict name->Polygon
    centroids = {name: (poly.centroid.y, poly.centroid.x) for name, poly in hulls.items()}

    # 5) Attribution mixte hull + fallback
    attribs = []
    for _, row in df_clients.iterrows():
        latc, lonc = row["Latitude"], row["Longitude"]
        choix = "HZ"
        if pd.notna(latc) and pd.notna(lonc):
            pt = Point(lonc, latc)
            # priorité hull contains/intersects
            for name, poly in hulls.items():
                if poly.intersects(pt):
                    choix = name
                    break
            else:
                # fallback par distance aux centroïdes
                dists = {
                    name: distance_haversine(latc, lonc, cy, cx)
                    for name, (cy, cx) in centroids.items()
                }
                nearest = min(dists, key=dists.get)
                if dists[nearest] <= FALLBACK_RADIUS_KM:
                    choix = nearest
        attribs.append(choix)

    df_clients["Tournée attribuée"] = attribs
    st.success("✅ Attribution terminée")

    # 6) Export Excel
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_clients.to_excel(writer, index=False)
    st.download_button("Télécharger le fichier enrichi (.xlsx)",
                       buffer.getvalue(),
                       file_name="clients_tournees_enrichi.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
