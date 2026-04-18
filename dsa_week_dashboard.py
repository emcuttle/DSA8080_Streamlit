# -----------------------------
# Imports
# -----------------------------
import streamlit as st
import pandas as pd
import pydeck as pdk
from google.cloud import bigquery
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import geopandas as gpd
from sklearn.cluster import DBSCAN
import json

# -----------------------------
# Title
# -----------------------------
st.title("Marshall CO Wildfire: Building Damage Statuses")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("DBSCAN Hotspot Detection")

enable_hotspots = st.sidebar.toggle("Show cluster hotspots", value=False)

intensity = st.sidebar.select_slider(
    "Hotspot intensity",
    options=["Low", "Medium", "High"],
    value="Medium",
)

# -----------------------------
# BigQuery
# -----------------------------
def get_bq_client():
    if "gcp_service_account" in st.secrets:
        creds_info = st.secrets["gcp_service_account"]
        return bigquery.Client.from_service_account_info(creds_info)
    return bigquery.Client()

@st.cache_data(ttl=300)
def get_bq_data():
    client = get_bq_client()
    query = """
        SELECT id, label, prediction_class, geometry
        FROM `capstone-project-485905.marshall_v9_seed_75.v_inference_results_geo`
    """
    return client.query(query).to_geodataframe()

# -----------------------------
# DBSCAN FUNCTION
# -----------------------------
@st.cache_data(ttl=300)
def run_dbscan(gdf, eps, min_samples, buffer_m):
    df = gdf.copy()

    if df.crs is None:
        df = df.set_crs(4326)

    if df.crs.is_geographic:
        proj = df.to_crs(df.estimate_utm_crs())
    else:
        proj = df

    centroids = proj.geometry.centroid

    y = pd.to_numeric(proj["prediction_class"], errors="coerce").fillna(0).astype(int)
    mask = y == 1

    labels = np.full(len(df), -1)

    hotspot_polys = gpd.GeoDataFrame(columns=["cluster", "geometry"], crs=proj.crs)

    circles = pd.DataFrame(columns=["lon", "lat", "radius_m"])

    if mask.sum() > 0:
        coords = np.column_stack([centroids[mask].x, centroids[mask].y])

        cluster_labels = DBSCAN(
            eps=eps,
            min_samples=min_samples
        ).fit_predict(coords)

        labels[mask.to_numpy()] = cluster_labels

        clustered = gpd.GeoDataFrame(
            {"cluster": cluster_labels},
            geometry=centroids[mask],
            crs=proj.crs
        )

        clustered = clustered[clustered["cluster"] != -1]

        if not clustered.empty:
            clustered["geometry"] = clustered.geometry.buffer(buffer_m)
            hotspot_polys = clustered.dissolve(by="cluster", as_index=False)

            c = hotspot_polys.geometry.centroid
            circles = pd.DataFrame({
                "lon": c.x,
                "lat": c.y,
                "radius_m": buffer_m
            })

    out = df.copy()
    out["db_cluster"] = labels
    out["is_cluster"] = labels != -1

    return out, hotspot_polys, circles

# -----------------------------
# Load data
# -----------------------------
gdf = get_bq_data()

gdf["prediction_class"] = pd.to_numeric(
    gdf["prediction_class"], errors="coerce"
).fillna(0).astype(int)

# -----------------------------
# Performance Metrics
# -----------------------------
with st.expander("Model Performance Metrics"):
    y_true = pd.to_numeric(gdf["label"], errors="coerce").fillna(0).astype(int)
    y_pred = gdf["prediction_class"]

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    st.caption("Confusion Matrix (Actual vs Predicted)")

# -----------------------------
# Intensity settings
# -----------------------------
params = {
    "Low": (650, 15, 750),
    "Medium": (350, 45, 450),
    "High": (250, 70, 300),
}

eps, min_s, buf = params[intensity]

hotspot_polys = None
circles = pd.DataFrame()

# -----------------------------
# RUN DBSCAN
# -----------------------------
if enable_hotspots:
    gdf, hotspot_polys, circles = run_dbscan(
        gdf,
        eps=eps,
        min_samples=min_s,
        buffer_m=buf
    )

    # -----------------------------
    # CLUSTER SUMMARY BOX (NEW)
    # -----------------------------
    num_clusters = len(hotspot_polys) if hotspot_polys is not None else 0
    num_hotpoints = int((gdf["db_cluster"] != -1).sum())

    st.sidebar.markdown("### Cluster Summary")
    st.sidebar.metric("Clusters Found", num_clusters)
    st.sidebar.metric("Buildings in Clusters", num_hotpoints)
    st.sidebar.caption(f"eps={eps}, min_samples={min_s}, buffer={buf}")

# -----------------------------
# Colors
# -----------------------------
gdf["fill_color"] = [
    [255, 69, 0, 220] if x == 1 else [0, 255, 255, 220]
    for x in gdf["prediction_class"]
]

# -----------------------------
# Layers (IMPORTANT FIX)
# -----------------------------
layers = []

# Buildings
layers.append(
    pdk.Layer(
        "GeoJsonLayer",
        gdf,
        get_fill_color="fill_color",
        opacity=0.85,
        pickable=True
    )
)

# Cluster polygons
if enable_hotspots and hotspot_polys is not None and not hotspot_polys.empty:
    layers.append(
        pdk.Layer(
            "GeoJsonLayer",
            json.loads(hotspot_polys.to_json()),
            filled=True,
            stroked=True,
            get_fill_color=[255, 140, 0, 120],
            get_line_color=[255, 80, 0, 255],
        )
    )

# Cluster circles
if enable_hotspots and not circles.empty:
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            circles,
            get_position=["lon", "lat"],
            get_radius="radius_m * 5",
            radius_units="meters",
            filled=True,
            stroked=True,
            get_fill_color=[255, 165, 0, 90],
            get_line_color=[255, 140, 0, 255],
            line_width_min_pixels=3,
        )
    )

# -----------------------------
# View
# -----------------------------
cent = gdf.geometry.centroid

view_state = pdk.ViewState(
    latitude=float(cent.y.mean()),
    longitude=float(cent.x.mean()),
    zoom=14,
    pitch=45,
)

st.pydeck_chart(
    pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip={"text": "ID: {id}\nLabel: {label}\nPred: {prediction_class}"}
    )
)

# -----------------------------
# Legend
# -----------------------------
st.markdown("""
### Legend
- 🔵 Cyan = Undamaged  
- 🔴 Red = Damaged  
- 🟠 Orange fill = DBSCAN clusters  
- 🟡 Orange rings = cluster boundaries  
""")
