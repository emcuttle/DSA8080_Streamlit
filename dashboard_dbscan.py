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
# Dashboard title
# -----------------------------
st.title("Marshall CO Wildfire: Building Damage Statuses")

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Cluster Hotspot Detection (DBSCAN)")
enable_hotspots = st.sidebar.toggle("Show cluster hotspots", value=False)

intensity = st.sidebar.select_slider(
    "Hotspot intensity",
    options=["Low", "Medium", "High"],
    value="Medium",
)

with st.sidebar.expander("What are cluster hotspots?"):
    st.write(
        "DBSCAN hotspots highlight dense clusters of predicted damaged buildings. "
        "Clusters are expanded using buffered centroids for visualization."
    )

if st.sidebar.button("Clear Streamlit caches"):
    st.cache_data.clear()
    st.sidebar.success("Caches cleared.")

# -----------------------------
# BigQuery connection
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
        SELECT 
            id, 
            label, 
            prediction_class, 
            geometry
        FROM `capstone-project-485905.marshall_v9_seed_75.v_inference_results_geo`
    """
    return client.query(query).to_geodataframe()

# -----------------------------
# DBSCAN hotspot computation
# -----------------------------
@st.cache_data(ttl=300, show_spinner="Computing DBSCAN hotspots…")
def add_dbscan_hotspots(
    _gdf,
    cache_key: str,
    damaged_col="prediction_class_num",
    damaged_value=1,
    eps_meters=350,
    min_samples=45,
    buffer_meters=450,
):
    gdf = _gdf.copy()

    if gdf.crs is None:
        gdf = gdf.set_crs(4326)

    if gdf.crs.is_geographic:
        gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())
    else:
        gdf_proj = gdf

    centroids = gdf_proj.geometry.centroid

    y = pd.to_numeric(gdf_proj[damaged_col], errors="coerce").fillna(0).astype(int)
    damaged_mask = (y == damaged_value)

    all_labels = np.full(len(gdf_proj), -1, dtype=int)

    hotspot_areas = gpd.GeoDataFrame(columns=["cluster", "geometry"], crs=gdf_proj.crs)

    if damaged_mask.sum() > 0:
        coords = np.column_stack([
            centroids[damaged_mask].x,
            centroids[damaged_mask].y
        ])

        labels = DBSCAN(
            eps=eps_meters,
            min_samples=min_samples
        ).fit_predict(coords)

        all_labels[damaged_mask.to_numpy()] = labels

        clustered = gpd.GeoDataFrame(
            {"cluster": labels},
            geometry=centroids[damaged_mask],
            crs=gdf_proj.crs
        )

        clustered = clustered[clustered["cluster"] != -1]

        if not clustered.empty:
            clustered["geometry"] = clustered.geometry.buffer(buffer_meters)
            hotspot_areas = clustered.dissolve(by="cluster", as_index=False)

    # convert to lat/lon
    hotspot_ll = hotspot_areas
    if not hotspot_ll.empty and not hotspot_ll.crs.is_geographic:
        hotspot_ll = hotspot_ll.to_crs(4326)

    # circles (centroids)
    circles_df = pd.DataFrame(columns=["lon", "lat", "radius_m"])
    if not hotspot_ll.empty:
        c = hotspot_ll.geometry.centroid
        circles_df = pd.DataFrame({
            "lon": c.x,
            "lat": c.y,
            "radius_m": buffer_meters
        })

    gdf_out = _gdf.copy()
    gdf_out["db_cluster"] = all_labels
    gdf_out["db_is_hotspot"] = all_labels != -1

    return gdf_out, hotspot_ll, circles_df


# -----------------------------
# Load data
# -----------------------------
gdf = get_bq_data()

gdf["prediction_class_num"] = pd.to_numeric(
    gdf["prediction_class"], errors="coerce"
).fillna(0).astype(int)

# -----------------------------
# Metrics section
# -----------------------------
with st.expander("Model Performance"):
    y_true = pd.to_numeric(gdf["label"], errors="coerce").fillna(0).astype(int)
    y_pred = gdf["prediction_class_num"]

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# -----------------------------
# DBSCAN parameters
# -----------------------------
params = {
    "Low": (650, 15, 750),
    "Medium": (350, 45, 450),
    "High": (250, 70, 300),
}

eps, min_s, buf = params[intensity]

hotspot_areas = None
circles_df = pd.DataFrame()

if enable_hotspots:
    gdf, hotspot_areas, circles_df = add_dbscan_hotspots(
        gdf,
        cache_key=f"{intensity}|{len(gdf)}",
        eps_meters=eps,
        min_samples=min_s,
        buffer_meters=buf
    )

# -----------------------------
# Colors
# -----------------------------
gdf["fill_color"] = [
    [255, 69, 0, 220] if x == 1 else [0, 255, 255, 220]
    for x in gdf["prediction_class_num"]
]

# -----------------------------
# Layers
# -----------------------------
building_layer = pdk.Layer(
    "GeoJsonLayer",
    gdf,
    get_fill_color="fill_color",
    pickable=True
)

layers = [building_layer]

# hotspot polygons
if enable_hotspots and hotspot_areas is not None and not hotspot_areas.empty:
    layers.append(
        pdk.Layer(
            "GeoJsonLayer",
            json.loads(hotspot_areas.to_json()),
            filled=True,
            stroked=True,
            get_fill_color=[255, 165, 0, 120],
            get_line_color=[255, 140, 0, 255],
        )
    )

# -----------------------------
# FIXED CIRCLE LAYER (IMPORTANT)
# -----------------------------
if enable_hotspots and not circles_df.empty:
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            circles_df,
            get_position=["lon", "lat"],

            # 🔥 FIX: scale radius so it is visible
            get_radius="radius_m * 3",

            radius_units="meters",

            stroked=True,
            filled=True,

            get_fill_color=[255, 165, 0, 80],
            get_line_color=[255, 140, 0, 255],

            line_width_min_pixels=2,
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
    pitch=30,
)

st.pydeck_chart(
    pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip={"text": "ID: {id}\nLabel: {label}\nPred: {prediction_class_num}"}
    )
)

# -----------------------------
# LEGEND (MANUAL)
# -----------------------------
st.markdown("""
### Map Legend
- 🔵 Cyan = Undamaged buildings  
- 🔴 Red = Damaged buildings  
- 🟠 Orange polygons = DBSCAN clusters  
- 🟡 Orange rings = cluster expansion zones  
""")
