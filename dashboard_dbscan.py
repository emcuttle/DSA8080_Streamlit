# -----------------------------
# Streamlit App (DBSCAN Hotspots - FULL COPY/PASTE, FIXED vars()/serialization)
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
    help=(
        "Low = more permissive (larger eps, smaller min_samples → bigger/more blobs)\n"
        "High = stricter (smaller eps, larger min_samples → fewer/tighter blobs)"
    ),
)

with st.sidebar.expander("What are cluster hotspots?"):
    st.write(
        "DBSCAN hotspots highlight dense clusters of predicted damaged buildings. "
        "We cluster damaged-building centroids, then buffer and dissolve clusters to create "
        "large hotspot blobs/circles around clusters."
    )

if st.sidebar.button("Clear Streamlit caches"):
    st.cache_data.clear()
    st.sidebar.success("Caches cleared. App will recompute on rerun.")

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
# DBSCAN hotspot computation (cached)
# IMPORTANT:
#  - _buildings_gdf has underscore so Streamlit doesn't hash it. [1](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_data)
#  - cache_key is hashable so changing intensity forces recompute. [1](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_data)
# -----------------------------
@st.cache_data(show_spinner="Computing DBSCAN hotspots…", ttl=300)
def add_dbscan_hotspots(
    _buildings_gdf: gpd.GeoDataFrame,
    cache_key: str,
    damaged_col: str = "prediction_class_num",
    damaged_value: int = 1,
    eps_meters: float = 350,
    min_samples: int = 45,
    buffer_meters: float = 450,
):
    gdf = _buildings_gdf.copy()

    # Ensure CRS
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)

    # Project to meters for DBSCAN/buffering
    if gdf.crs.is_geographic:
        gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())
    else:
        gdf_proj = gdf

    # Centroids for clustering
    centroids = gdf_proj.geometry.centroid

    # Damaged mask (force numeric)
    y = pd.to_numeric(gdf_proj[damaged_col], errors="coerce").fillna(0).astype(int)
    damaged_mask = (y == damaged_value)
    damaged_count = int(damaged_mask.sum())

    # Default labels = noise
    all_labels = np.full(len(gdf_proj), -1, dtype=int)

    # Hotspot blobs (cluster polygons)
    hotspot_areas = gpd.GeoDataFrame({"cluster": [], "geometry": []}, crs=gdf_proj.crs)
    cluster_count = 0

    if damaged_count > 0:
        coords = np.column_stack([
            centroids[damaged_mask].x.to_numpy(),
            centroids[damaged_mask].y.to_numpy()
        ])

        labels = DBSCAN(eps=eps_meters, min_samples=min_samples, metric="euclidean").fit_predict(coords)
        all_labels[damaged_mask.to_numpy()] = labels

        unique = set(labels.tolist())
        cluster_count = len([c for c in unique if c != -1])

        clustered_pts = gpd.GeoDataFrame(
            {"cluster": labels},
            geometry=centroids[damaged_mask],
            crs=gdf_proj.crs
        )
        clustered_pts = clustered_pts[clustered_pts["cluster"] != -1].copy()

        if not clustered_pts.empty:
            clustered_pts["geometry"] = clustered_pts.geometry.buffer(buffer_meters)
            hotspot_areas = clustered_pts.dissolve(by="cluster", as_index=False)[["cluster", "geometry"]]

    # Attach labels back to original
    buildings_out = gdf.copy()
    buildings_out["db_cluster"] = all_labels
    buildings_out["db_is_hotspot"] = buildings_out["db_cluster"] != -1

    # Convert hotspot polygons back to lat/lon for mapping
    hotspot_areas_ll = hotspot_areas
    if hotspot_areas_ll is not None and not hotspot_areas_ll.empty:
        if not hotspot_areas_ll.crs.is_geographic:
            hotspot_areas_ll = hotspot_areas_ll.to_crs(4326)

    # IMPORTANT FIX:
    # Build circle centers as a *plain pandas DataFrame* (no geometry objects),
    # so ScatterplotLayer doesn't attempt to serialize shapely geometries.
    circles_df = pd.DataFrame(columns=["lon", "lat", "radius_m"])
    if hotspot_areas_ll is not None and not hotspot_areas_ll.empty:
        cent = hotspot_areas_ll.geometry.centroid
        circles_df = pd.DataFrame({
            "lon": cent.x.to_numpy(),
            "lat": cent.y.to_numpy(),
            "radius_m": np.full(len(hotspot_areas_ll), float(buffer_meters))
        })

    debug = {
        "damaged_count": damaged_count,
        "cluster_count": cluster_count,
        "hotspot_polygon_count": int(0 if hotspot_areas_ll is None else len(hotspot_areas_ll)),
    }

    return buildings_out, hotspot_areas_ll, circles_df, debug

# -----------------------------
# Main
# -----------------------------
try:
    gdf = get_bq_data()

    # Normalize prediction column to int
    gdf["prediction_class_num"] = pd.to_numeric(gdf["prediction_class"], errors="coerce").fillna(0).astype(int)
    pred_damaged_count = int((gdf["prediction_class_num"] == 1).sum())
    st.sidebar.caption(f"Predicted damaged buildings: {pred_damaged_count}")

    # -----------------------------
    # Model Performance
    # -----------------------------
    with st.expander("View Model Performance Metrics"):
        col1, col2 = st.columns(2)

        with col1:
            st.write("Confusion Matrix")
            y_true = pd.to_numeric(gdf["label"], errors="coerce").fillna(0).astype(int)
            y_pred = gdf["prediction_class_num"].astype(int)
            cm = confusion_matrix(y_true, y_pred)

            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Undamaged", "Damaged"],
                yticklabels=["Undamaged", "Damaged"]
            )
            ax.set_ylabel("Actual")
            ax.set_xlabel("Predicted")
            st.pyplot(fig)

        with col2:
            st.write("Prediction Distribution")
            fig2, ax2 = plt.subplots(figsize=(4, 3))
            sns.countplot(
                x=gdf["prediction_class_num"],
                ax=ax2,
                palette=["#00FFFF", "#FF4500"],
                order=[0, 1]
            )
            ax2.set_xticklabels(["Undamaged", "Damaged"])
            ax2.set_xlabel("Status")
            ax2.set_ylabel("Count")
            st.pyplot(fig2)

    # -----------------------------
    # Intensity parameters (more separated; avoid High=0)
    # -----------------------------
    intensity_params = {
        "Low":    {"eps": 650, "min_samples": 15,  "buffer": 750},
        "Medium": {"eps": 350, "min_samples": 45,  "buffer": 450},
        "High":   {"eps": 250, "min_samples": 70,  "buffer": 300},
    }
    params = intensity_params[intensity]

    hotspot_areas = None
    circles_df = pd.DataFrame(columns=["lon", "lat", "radius_m"])
    debug = {"damaged_count": 0, "cluster_count": 0, "hotspot_polygon_count": 0}

    if enable_hotspots:
        cache_key = f"{intensity}|eps={params['eps']}|min={params['min_samples']}|buf={params['buffer']}|rows={len(gdf)}"

        gdf, hotspot_areas, circles_df, debug = add_dbscan_hotspots(
            gdf,
            cache_key=cache_key,
            damaged_col="prediction_class_num",
            damaged_value=1,
            eps_meters=float(params["eps"]),
            min_samples=int(params["min_samples"]),
            buffer_meters=float(params["buffer"]),
        )

        st.sidebar.caption(f"DBSCAN clusters found: {debug['cluster_count']}")
        st.sidebar.caption(f"Hotspot polygons: {debug['hotspot_polygon_count']}")

    # -----------------------------
    # Building fill colors (safe list-of-lists)
    # -----------------------------
    preds = gdf["prediction_class_num"].astype(int).tolist()
    gdf["fill_color"] = [
        [255, 69, 0, 220] if v == 1 else [0, 255, 255, 220]
        for v in preds
    ]

    tooltip_html = (
        "<b>Building ID:</b> {id}<br>"
        "<b>Actual Label:</b> {label}<br>"
        "<b>Prediction:</b> {prediction_class_num}<br>"
        "<b>DBSCAN cluster:</b> {db_cluster}<br>"
        "<b>In hotspot?:</b> {db_is_hotspot}"
    )

    # -----------------------------
    # Layers
    # -----------------------------
    building_layer = pdk.Layer(
        "GeoJsonLayer",
        gdf,
        opacity=0.85,
        stroked=False,
        filled=True,
        get_fill_color="fill_color",
        pickable=True,
    )

    layers = [building_layer]

    # --- Blob polygons (buffer+dissolve)
    if enable_hotspots and hotspot_areas is not None and not hotspot_areas.empty:
        hotspot_geojson = json.loads(hotspot_areas.to_json())

        hotspot_blob_layer = pdk.Layer(
            "GeoJsonLayer",
            hotspot_geojson,
            opacity=0.65,
            stroked=True,
            filled=True,
            get_fill_color=[255, 165, 0, 160],   # orange fill
            get_line_color=[255, 120, 0, 255],   # darker outline
            # outline visibility controls [2](https://alasarr.github.io/deck.gl/docs/api-reference/layers/geojson-layer)[3](https://deck.gl/docs/api-reference/layers/geojson-layer)
            get_line_width=1,
            line_width_scale=10,
            line_width_min_pixels=3,
            pickable=True,
        )
        layers.append(hotspot_blob_layer)

    # --- Big circles at cluster centroids (plain DataFrame -> avoids vars() serialization error)
    if enable_hotspots and circles_df is not None and not circles_df.empty:
        circle_layer = pdk.Layer(
            "ScatterplotLayer",
            circles_df,
            get_position=["lon", "lat"],
            get_radius="radius_m",
            radius_units="meters",
            stroked=True,
            filled=False,
            get_line_color=[255, 255, 0, 255],  # bright yellow rings
            line_width_min_pixels=3,
            pickable=False,
        )
        layers.append(circle_layer)

    # -----------------------------
    # View state
    # -----------------------------
    gdf_ll = gdf
    if gdf_ll.crs is None:
        gdf_ll = gdf_ll.set_crs(4326)
    if not gdf_ll.crs.is_geographic:
        gdf_ll = gdf_ll.to_crs(4326)

    view_state = pdk.ViewState(
        latitude=float(gdf_ll.geometry.centroid.y.mean()),
        longitude=float(gdf_ll.geometry.centroid.x.mean()),
        zoom=14,
        pitch=30,
    )

    st.pydeck_chart(
        pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip={"html": tooltip_html},
        )
    )

except Exception as e:
    st.error(f"App error: {e}")
    st.info("If this happened after code changes, try the 'Clear Streamlit caches' button in the sidebar.")
