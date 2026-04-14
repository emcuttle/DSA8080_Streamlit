# -----------------------------
# Streamlit App (DBSCAN Hotspots - FULL FIX)
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
# Sidebar controls
# -----------------------------
st.sidebar.header("Cluster Hotspot Detection (DBSCAN)")
enable_hotspots = st.sidebar.toggle("Show cluster hotspots", value=False)

intensity = st.sidebar.select_slider(
    "Hotspot intensity",
    options=["Low", "Medium", "High"],
    value="Medium",
    help=(
        "Low = more permissive (larger eps, smaller min_samples -> more/bigger blobs)\n"
        "High = stricter (smaller eps, larger min_samples -> fewer/tighter blobs)"
    ),
)

with st.sidebar.expander("What are cluster hotspots?"):
    st.write(
        "DBSCAN hotspots highlight dense clusters of predicted damaged buildings. "
        "We cluster damaged building centroids, then buffer and dissolve them to create "
        "large hotspot blobs around clusters."
    )

# Optional: easy cache reset button while debugging deployments
if st.sidebar.button("Clear Streamlit caches"):
    st.cache_data.clear()
    st.sidebar.success("Caches cleared. Rerun will recompute.")

# -----------------------------
# BigQuery client + data pull
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
# IMPORTANT:
# - _buildings_gdf has a leading underscore so Streamlit does NOT try to hash it. [1](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_data)
# - cache_key is a hashable string that forces recompute when intensity/params change. [1](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_data)
# -----------------------------
@st.cache_data(show_spinner="Computing DBSCAN cluster hotspots…", ttl=300)
def add_dbscan_hotspots_v2(
    _buildings_gdf: gpd.GeoDataFrame,  # <-- excluded from hashing by Streamlit [1](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_data)
    cache_key: str,                   # <-- hashed; changing this forces recompute [1](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_data)
    damaged_col: str = "prediction_class_num",
    damaged_value: int = 1,
    eps_meters: float = 300,
    min_samples: int = 50,
    buffer_meters: float = 350,
):
    gdf = _buildings_gdf.copy()

    # Ensure CRS
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)

    # Project to meters for DBSCAN eps/buffer
    if gdf.crs.is_geographic:
        gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())
    else:
        gdf_proj = gdf

    # Centroids for clustering
    centroids = gdf_proj.geometry.centroid

    # Build damaged mask
    y = pd.to_numeric(gdf_proj[damaged_col], errors="coerce").fillna(0).astype(int)
    damaged_mask = (y == damaged_value)
    damaged_count = int(damaged_mask.sum())

    # Default labels = noise
    all_labels = np.full(len(gdf_proj), -1, dtype=int)

    # Placeholder hotspot polygons
    hotspot_areas = gpd.GeoDataFrame({"cluster": [], "geometry": []}, crs=gdf_proj.crs)
    cluster_count = 0

    if damaged_count > 0:
        coords = np.column_stack([
            centroids[damaged_mask].x.to_numpy(),
            centroids[damaged_mask].y.to_numpy()
        ])

        labels = DBSCAN(eps=eps_meters, min_samples=min_samples, metric="euclidean").fit_predict(coords)
        all_labels[damaged_mask.to_numpy()] = labels

        # count clusters excluding noise (-1)
        unique = set(labels.tolist())
        cluster_count = len([c for c in unique if c != -1])

        clustered_pts = gpd.GeoDataFrame(
            {"cluster": labels},
            geometry=centroids[damaged_mask],
            crs=gdf_proj.crs
        )
        clustered_pts = clustered_pts[clustered_pts["cluster"] != -1].copy()

        # Buffer + dissolve -> big blobs
        if not clustered_pts.empty:
            clustered_pts["geometry"] = clustered_pts.geometry.buffer(buffer_meters)
            hotspot_areas = clustered_pts.dissolve(by="cluster", as_index=False)[["cluster", "geometry"]]

    # Attach cluster labels back to original (same row order)
    buildings_out = gdf.copy()
    buildings_out["db_cluster"] = all_labels
    buildings_out["db_is_hotspot"] = buildings_out["db_cluster"] != -1

    # Convert hotspot polygons to lat/lon for mapping
    hotspot_areas_ll = hotspot_areas
    if hotspot_areas_ll is not None and not hotspot_areas_ll.empty:
        if not hotspot_areas_ll.crs.is_geographic:
            hotspot_areas_ll = hotspot_areas_ll.to_crs(4326)

    debug = {
        "damaged_count": damaged_count,
        "cluster_count": cluster_count,
        "hotspot_polygon_count": int(0 if hotspot_areas_ll is None else len(hotspot_areas_ll)),
    }

    return buildings_out, hotspot_areas_ll, debug

# -----------------------------
# Main execution
# -----------------------------
try:
    gdf = get_bq_data()

    # Normalize prediction column
    gdf["prediction_class_num"] = pd.to_numeric(gdf["prediction_class"], errors="coerce").fillna(0).astype(int)
    pred_damaged_count = int((gdf["prediction_class_num"] == 1).sum())
    st.sidebar.caption(f"Predicted damaged buildings: {pred_damaged_count}")

    # -----------------------------
    # Model performance expander
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
    # Intensity -> parameters (tuned for ~2000 damaged)
    # -----------------------------
    intensity_params = {
        "Low":    {"eps": 500, "min_samples": 25,  "buffer": 550},  # broad blobs
        "Medium": {"eps": 300, "min_samples": 60,  "buffer": 350},
        "High":   {"eps": 180, "min_samples": 140, "buffer": 250},  # tight/dense blobs
    }
    params = intensity_params[intensity]

    hotspot_areas = None
    debug = {"damaged_count": 0, "cluster_count": 0, "hotspot_polygon_count": 0}

    # -----------------------------
    # Hotspot logic
    # -----------------------------
    if enable_hotspots:
        # cache_key forces recompute when intensity changes (and when data size changes)
        cache_key = f"{intensity}|eps={params['eps']}|min={params['min_samples']}|buf={params['buffer']}|rows={len(gdf)}"

        gdf, hotspot_areas, debug = add_dbscan_hotspots_v2(
            gdf,  # positional goes into _buildings_gdf (unhashed) [1](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_data)
            cache_key=cache_key,
            damaged_col="prediction_class_num",
            damaged_value=1,
            eps_meters=float(params["eps"]),
            min_samples=int(params["min_samples"]),
            buffer_meters=float(params["buffer"]),
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
    else:
        tooltip_html = (
            "<b>Building ID:</b> {id}<br>"
            "<b>Actual Label:</b> {label}<br>"
            "<b>Prediction:</b> {prediction_class_num}"
        )

    # -----------------------------
    # Colors for buildings
    # -----------------------------
    def fill_color(row):
        return [255, 69, 0] if int(row["prediction_class_num"]) == 1 else [0, 255, 255]

    gdf["fill_color"] = gdf.apply(fill_color, axis=1)

    # -----------------------------
    # Build layers
    # -----------------------------
    building_layer = pdk.Layer(
        "GeoJsonLayer",
        gdf,
        opacity=0.9,
        stroked=False,
        filled=True,
        get_fill_color="fill_color",
        pickable=True,
    )

    layers = [building_layer]

    # Hotspot overlay blobs
    if enable_hotspots and hotspot_areas is not None and not hotspot_areas.empty:
        hotspot_geojson = json.loads(hotspot_areas.to_json())

        hotspot_layer = pdk.Layer(
            "GeoJsonLayer",
            hotspot_geojson,
            opacity=0.95,
            stroked=True,
            filled=True,
            get_fill_color=[255, 165, 0, 150],   # orange fill
            get_line_color=[255, 140, 0, 255],   # orange outline
            get_line_width=10,
            line_width_min_pixels=2,             # make outline visible
            line_width_scale=1,
            pickable=True,
        )
        layers.append(hotspot_layer)

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
    st.info("Tip: If you just changed code on Streamlit Cloud, try 'Clear Streamlit caches' in the sidebar and rerun.")
