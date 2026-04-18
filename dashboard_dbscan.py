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
# Data
# -----------------------------
def get_bq_client():
    if "gcp_service_account" in st.secrets:
        return bigquery.Client.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
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
# DBSCAN (FIXED PROJECTION + GUARANTEED OUTPUT)
# -----------------------------
@st.cache_data(ttl=300)
def run_dbscan(_gdf, eps, min_samples, buffer_m):

    df = _gdf.copy()

    if df.crs is None:
        df = df.set_crs(4326)

    # 🔥 CRITICAL FIX: force projection BEFORE clustering
    proj = df.to_crs(df.estimate_utm_crs())

    centroids = proj.geometry.centroid

    y = pd.to_numeric(proj["prediction_class"], errors="coerce").fillna(0).astype(int)
    mask = y == 1

    labels = np.full(len(df), -1)

    circles = pd.DataFrame(columns=["lon", "lat"])

    if mask.sum() > 0:

        coords = np.column_stack([centroids[mask].x, centroids[mask].y])

        # 🔥 DBSCAN clustering
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

        # 🔥 IMPORTANT: ALWAYS return centroid points (NOT just polygons)
        if not clustered.empty:
            c_ll = clustered.to_crs(4326)
            circles = pd.DataFrame({
                "lon": c_ll.geometry.x,
                "lat": c_ll.geometry.y
            })

    out = df.copy()
    out["db_cluster"] = labels
    out["is_cluster"] = labels != -1

    return out, circles

# -----------------------------
# Load data
# -----------------------------
gdf = get_bq_data()

gdf["prediction_class"] = pd.to_numeric(
    gdf["prediction_class"], errors="coerce"
).fillna(0).astype(int)

# -----------------------------
# Intensity tuning (now MATTERS)
# -----------------------------
params = {
    "Low": (900, 10, 0),
    "Medium": (600, 20, 0),
    "High": (350, 30, 0),
}

eps, min_s, buf = params[intensity]

circles = pd.DataFrame()

if enable_hotspots:
    gdf, circles = run_dbscan(
        gdf,
        eps=eps,
        min_samples=min_s,
        buffer_m=buf
    )

    st.sidebar.markdown("### Cluster Summary")
    st.sidebar.metric("Clusters Found", len(set(gdf["db_cluster"])) - 1)
    st.sidebar.metric("Clustered Buildings", int((gdf["db_cluster"] != -1).sum()))

# -----------------------------
# Colors (fixed)
# -----------------------------
gdf["fill_color"] = [
    [0, 195, 255, 220] if x == 0 else [255, 59, 48, 220]
    for x in gdf["prediction_class"]
]

# -----------------------------
# LEGEND (Gi*-STYLE ABOVE MAP)
# -----------------------------
st.markdown("""
<div style="display:flex; gap:25px; margin-bottom:10px; font-weight:bold;">

  <div style="display:flex; align-items:center; gap:6px;">
    <div style="width:12px;height:12px;background:rgb(0,195,255);border-radius:50%"></div>
    Undamaged
  </div>

  <div style="display:flex; align-items:center; gap:6px;">
    <div style="width:12px;height:12px;background:rgb(255,59,48);border-radius:50%"></div>
    Damaged
  </div>

  <div style="display:flex; align-items:center; gap:6px;">
    <div style="width:12px;height:12px;background:orange;border-radius:50%"></div>
    DBSCAN clusters
  </div>

</div>
""", unsafe_allow_html=True)

# -----------------------------
# LAYERS (CRITICAL FIX)
# -----------------------------
layers = []

# buildings
layers.append(
    pdk.Layer(
        "GeoJsonLayer",
        gdf,
        get_fill_color="fill_color",
        opacity=0.85,
        pickable=True
    )
)

# 🔥 DBSCAN POINT VISUALIZATION (THIS FIXES YOUR “NO CLUSTERS” ISSUE)
if enable_hotspots and not circles.empty:
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            circles,
            get_position=["lon", "lat"],
            get_radius=200,   # fixed visible radius
            radius_units="meters",
            filled=True,
            get_fill_color=[255, 165, 0, 180],
            pickable=False
        )
    )

# -----------------------------
# VIEW
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


st.write("Unique DBSCAN labels:", np.unique(gdf["db_cluster"]))
st.write("Cluster count:", len(set(gdf["db_cluster"])) - (1 if -1 in gdf["db_cluster"] else 0))
st.write("Clustered points:", (gdf["db_cluster"] != -1).sum())
