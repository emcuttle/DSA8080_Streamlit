# -----------------------------
# Streamlit App (DBSCAN Hotspots - FIXED)
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
# Hotspot Toggle + Intensity Slider
# -----------------------------
st.sidebar.header("Cluster Hotspot Detection (DBSCAN)")
enable_hotspots = st.sidebar.toggle("Show cluster hotspots", value=False)

intensity = st.sidebar.select_slider(
    "Hotspot intensity",
    options=["Low", "Medium", "High"],
    value="Medium",
    help=(
        "Low = more permissive (larger eps, smaller min_samples => more/bigger hotspots)\n"
        "High = stricter (smaller eps, larger min_samples => fewer/tighter hotspots)"
    ),
)

with st.sidebar.expander("What are cluster hotspots?"):
    st.write(
        "DBSCAN hotspots highlight dense clusters of predicted damaged buildings. "
        "We cluster damaged-building centroids, then draw large buffered polygons around each cluster "
        "to create the 'large circles/blobs' your group prefers."
    )

# -----------------------------
# BigQuery Connection
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
# DBSCAN Hotspot Code (with cache_key to force recompute)
# -----------------------------
@st.cache_data(show_spinner="Computing DBSCAN cluster hotspots…", ttl=300)
def add_dbscan_hotspots(
    buildings_gdf: gpd.GeoDataFrame,
    cache_key: str,  # <-- IMPORTANT: include intensity/params so cache doesn't reuse old result
    damaged_col: str = "prediction_class",
    damaged_value: int = 1,
    eps_meters: float = 250,
    min_samples: int = 25,
    buffer_meters: float = 300
):
    """
    DBSCAN cluster hotspots for predicted damaged buildings.

    Returns:
      - buildings_out: original polygons with dbscan columns added
      - hotspot_areas_ll: dissolved hotspot polygons in EPSG:4326 (for mapping)
      - debug: dict of counts for sidebar
    """

    gdf = buildings_gdf.copy()

    # Ensure CRS exists (assume EPSG:4326 if missing)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)

    # Project to a CRS in meters if currently geographic (degrees)
    if gdf.crs.is_geographic:
        gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())
    else:
        gdf_proj = gdf

    # Centroids as a GeoSeries (so .x and .y work reliably)
    centroids = gdf_proj.geometry.centroid

    # Damaged mask (force numeric)
    y = pd.to_numeric(gdf_proj[damaged_col], errors="coerce").fillna(0).astype(int)
    damaged_mask = (y == damaged_value)
    damaged_count = int(damaged_mask.sum())

    # Initialize everything as noise (-1)
    all_labels = np.full(len(gdf_proj), -1, dtype=int)
    hotspot_areas = gpd.GeoDataFrame({"cluster": [], "geometry": []}, crs=gdf_proj.crs)

    cluster_count = 0

    if damaged_count > 0:
        coords = np.column_stack([centroids[damaged_mask].x.to_numpy(),
                                  centroids[damaged_mask].y.to_numpy()])

        # Run DBSCAN
        clusterer = DBSCAN(eps=eps_meters, min_samples=min_samples, metric="euclidean")
        labels = clusterer.fit_predict(coords)

        # write labels back
        all_labels[damaged_mask.to_numpy()] = labels

        # Count clusters (exclude noise=-1)
        unique = set(labels.tolist())
        cluster_count = len([c for c in unique if c != -1])

        # build hotspot polygons (buffer + dissolve) for clustered points
        clustered_pts = gpd.GeoDataFrame(
            {"cluster": labels},
            geometry=centroids[damaged_mask],
            crs=gdf_proj.crs
        )
        clustered_pts = clustered_pts[clustered_pts["cluster"] != -1].copy()

        if not clustered_pts.empty:
            clustered_pts["geometry"] = clustered_pts.geometry.buffer(buffer_meters)
            hotspot_areas = clustered_pts.dissolve(by="cluster", as_index=False)[["cluster", "geometry"]]

    # Attach labels back to original (unprojected) gdf (same row order)
    buildings_out = gdf.copy()
    buildings_out["db_cluster"] = all_labels
    buildings_out["db_is_hotspot"] = buildings_out["db_cluster"] != -1

    # Convert hotspot polygons back to EPSG:4326 for pydeck
    hotspot_areas_ll = hotspot_areas
    if hotspot_areas_ll is not None and not hotspot_areas_ll.empty:
        if hotspot_areas_ll.crs is None:
            hotspot_areas_ll = hotspot_areas_ll.set_crs(4326)
        elif not hotspot_areas_ll.crs.is_geographic:
            hotspot_areas_ll = hotspot_areas_ll.to_crs(4326)

    debug = {
        "damaged_count": damaged_count,
        "cluster_count": cluster_count,
        "hotspot_polygon_count": int(0 if hotspot_areas_ll is None else len(hotspot_areas_ll))
    }

    return buildings_out, hotspot_areas_ll, debug

# -----------------------------
# Main App
# -----------------------------
try:
    gdf = get_bq_data()

    # Quick diagnostics: how many predicted damaged?
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
            y_true = gdf["label"].astype(str).str.strip().astype(int)
            y_pred = gdf["prediction_class"].astype(str).str.strip().astype(int)
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["Undamaged", "Damaged"],
                        yticklabels=["Undamaged", "Damaged"])
            plt.ylabel("Actual")
            plt.xlabel("Predicted")
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
    # Intensity -> DBSCAN Params (tuned for large N)
    # -----------------------------
    intensity_params = {
        # With ~2000 damaged, these should visibly change results.
        "Low":    {"eps": 450, "min_samples": 20,  "buffer": 500},  # more/bigger blobs
        "Medium": {"eps": 300, "min_samples": 50,  "buffer": 350},
        "High":   {"eps": 180, "min_samples": 120, "buffer": 250},  # fewer/tighter blobs
    }
    params = intensity_params[intensity]

    hotspot_areas = None
    debug = {"damaged_count": 0, "cluster_count": 0, "hotspot_polygon_count": 0}

    # -----------------------------
    # Hotspot computation (DBSCAN)
    # -----------------------------
    if enable_hotspots:
        # cache_key ensures recompute when intensity/params change
        cache_key = f"{intensity}|eps={params['eps']}|min={params['min_samples']}|buf={params['buffer']}"

        gdf, hotspot_areas, debug = add_dbscan_hotspots(
            buildings_gdf=gdf,
            cache_key=cache_key,
            damaged_col="prediction_class_num",
            damaged_value=1,
            eps_meters=float(params["eps"]),
            min_samples=int(params["min_samples"]),
            buffer_meters=float(params["buffer"])
        )

        st.sidebar.caption(f"DBSCAN clusters found: {debug['cluster_count']}")
        st.sidebar.caption(f"Hotspot polygons: {debug['hotspot_polygon_count']}")

        st.markdown(
            f"""
            <div style="margin-bottom:10px; font-weight:bold;">DBSCAN Hotspot Legend</div>
            <div style="margin-bottom:6px;">
              <span style="font-weight:bold;">Intensity:</span> {intensity}
              &nbsp;&nbsp;|&nbsp;&nbsp;
              <span style="font-weight:bold;">eps:</span> {params["eps"]} m
              &nbsp;&nbsp;|&nbsp;&nbsp;
              <span style="font-weight:bold;">min_samples:</span> {params["min_samples"]}
              &nbsp;&nbsp;|&nbsp;&nbsp;
              <span style="font-weight:bold;">buffer:</span> {params["buffer"]} m
            </div>
            <div style="display:flex; gap:20px; align-items:center; margin-bottom: 10px;">
              <div style="width:20px; height:20px; background-color: rgb(0,255,255); border:1px solid white;"></div>
              <span>Undamaged (prediction)</span>
              <div style="width:20px; height:20px; background-color: rgb(255,69,0); border:1px solid white;"></div>
              <span>Damaged (prediction)</span>
              <div style="width:20px; height:20px; background-color: rgba(255,165,0,0.55); border:3px solid orange;"></div>
              <span>DBSCAN hotspot area</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        tooltip_html = (
            "<b>Building ID:</b> {id}<br>"
            "<b>Actual Label:</b> {label}<br>"
            "<b>Prediction:</b> {prediction_class_num}<br>"
            "<b>DBSCAN cluster:</b> {db_cluster}<br>"
            "<b>In hotspot?:</b> {db_is_hotspot}"
        )

        # Fill buildings by prediction
        gdf["fill_color"] = np.where(gdf["prediction_class_num"] == 1,
                                     [[255, 69, 0]] * len(gdf),
                                     [[0, 255, 255]] * len(gdf))
        get_fill_color = "fill_color"

    else:
        st.markdown(
            """
            <div style="display: flex; gap: 20px; align-items: center; margin-bottom: 10px; font-weight: bold;">
              <div style="width: 20px; height: 20px; background-color: rgb(0, 255, 255); border: 1px solid white;"></div>
              <span>Undamaged</span>
              <div style="width: 20px; height: 20px; background-color: rgb(255, 69, 0); border: 1px solid white;"></div>
              <span>Damaged</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        tooltip_html = (
            "<b>Building ID:</b> {id}<br>"
            "<b>Actual Label:</b> {label}<br>"
            "<b>Prediction:</b> {prediction_class_num}"
        )
        get_fill_color = "prediction_class_num == 1 ? [255, 69, 0] : [0, 255, 255]"

    # -----------------------------
    # Build Map Layers
    # -----------------------------
    building_layer = pdk.Layer(
        "GeoJsonLayer",
        gdf,
        opacity=0.9,
        stroked=False,
        filled=True,
        get_fill_color=get_fill_color,
        pickable=True,
    )

    layers = [building_layer]

    # Hotspot overlay layer (large circles/blobs)
    if enable_hotspots and hotspot_areas is not None and not hotspot_areas.empty:
        # Convert to GeoJSON dict to avoid serialization edge cases
        hotspot_geojson = json.loads(hotspot_areas.to_json())

        hotspot_layer = pdk.Layer(
            "GeoJsonLayer",
            hotspot_geojson,
            opacity=0.95,
            stroked=True,
            filled=True,
            get_fill_color=[255, 165, 0, 160],   # visible orange fill
            get_line_color=[255, 140, 0, 255],   # visible outline
            get_line_width=10,
            line_width_min_pixels=2,             # ensures outline is visible
            line_width_scale=1,
            pickable=True,
        )
        layers.append(hotspot_layer)

    # Ensure lat/lon for view state
    gdf_ll = gdf
    if gdf_ll.crs is None:
        gdf_ll = gdf_ll.set_crs(4326)
    if not gdf_ll.crs.is_geographic:
        gdf_ll = gdf_ll.to_crs(4326)

    view_state = pdk.ViewState(
        latitude=float(gdf_ll.geometry.centroid.y.mean()),
        longitude=float(gdf_ll.geometry.centroid.x.mean()),
        zoom=14,
        pitch=45,
    )

    st.pydeck_chart(
        pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip={"html": tooltip_html}
        )
    )

except Exception as e:
    st.error(f"App error: {e}")
    st.info("If this happened after turning on hotspots, it may be a geometry/CRS issue or missing dependency.")
