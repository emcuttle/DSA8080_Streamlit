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
        "Controls DBSCAN sensitivity:\n"
        "- Low: more permissive (larger eps, smaller min cluster size)\n"
        "- High: stricter (smaller eps, larger min cluster size)"
    ),
)

with st.sidebar.expander("What are cluster hotspots?"):
    st.write(
        "DBSCAN hotspots highlight dense clusters of predicted damaged buildings. "
        "We cluster damaged-building centroids, then draw large buffered polygons around each cluster "
        "to create the large 'circle/blob' hotspot areas."
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
# DBSCAN Hotspot Code
# -----------------------------
@st.cache_data(show_spinner="Computing DBSCAN cluster hotspots…", ttl=300)
def add_dbscan_hotspots(_buildings_gdf: gpd.GeoDataFrame,
                        damaged_col: str = "prediction_class",
                        damaged_value: int = 1,
                        eps_meters: float = 250,
                        min_samples: int = 75,
                        buffer_meters: float = 250):
    """
    DBSCAN cluster hotspots for predicted damaged buildings.

    Steps:
      1) Project to a local CRS in meters (UTM) so eps/buffer are meters.
      2) Cluster damaged building centroids with DBSCAN.
      3) Buffer clustered points and dissolve by cluster id to create hotspot polygons.

    Returns:
      - buildings_out: original polygons with dbscan columns added
      - hotspot_areas_ll: dissolved hotspot polygons in EPSG:4326 (for mapping)
    """

    gdf = _buildings_gdf.copy()

    # Ensure CRS exists (assume EPSG:4326 if missing)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)

    # Project to UTM (meters) if currently geographic (degrees)
    if gdf.crs.is_geographic:
        gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())
    else:
        gdf_proj = gdf

    # Use centroids for clustering
    pts = gdf_proj.copy()
    pts["centroid"] = pts.geometry.centroid

    # Damaged mask (force numeric)
    y = pd.to_numeric(pts[damaged_col], errors="coerce").fillna(0).astype(int)
    damaged_mask = (y == damaged_value)

    # Initialize everything as noise (-1)
    all_labels = np.full(len(pts), -1, dtype=int)
    hotspot_areas = gpd.GeoDataFrame({"cluster": [], "geometry": []}, crs=gdf_proj.crs)

    # Only attempt clustering if enough damaged points exist
    if damaged_mask.sum() >= min_samples:
        coords = np.column_stack([
            pts.loc[damaged_mask, "centroid"].x.to_numpy(),
            pts.loc[damaged_mask, "centroid"].y.to_numpy()
        ])

        # DBSCAN: eps is neighborhood radius; min_samples is min points to form dense region
        clusterer = DBSCAN(eps=eps_meters, min_samples=min_samples, metric="euclidean")
        labels = clusterer.fit_predict(coords)

        # Write labels back into full array
        all_labels[damaged_mask.to_numpy()] = labels

        # Build hotspot polygons (exclude noise = -1)
        clustered_pts = gpd.GeoDataFrame(
            {"cluster": labels},
            geometry=pts.loc[damaged_mask, "centroid"],
            crs=gdf_proj.crs
        )
        clustered_pts = clustered_pts[clustered_pts["cluster"] != -1].copy()

        if not clustered_pts.empty:
            clustered_pts["geometry"] = clustered_pts.geometry.buffer(buffer_meters)
            hotspot_areas = clustered_pts.dissolve(by="cluster", as_index=False)[["cluster", "geometry"]]

    # Attach labels to ORIGINAL polygon rows (same order)
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

    return buildings_out, hotspot_areas_ll

# -----------------------------
# Main App
# -----------------------------
try:
    gdf = get_bq_data()

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
                x=gdf["prediction_class"].astype(int),
                ax=ax2,
                palette=["#00FFFF", "#FF4500"],
                order=[0, 1]
            )
            ax2.set_xticklabels(["Undamaged", "Damaged"])
            ax2.set_xlabel("Status")
            ax2.set_ylabel("Count")
            st.pyplot(fig2)

    # -----------------------------
    # Intensity -> DBSCAN Params
    # -----------------------------
    # NOTE: High intensity = stricter (tighter, fewer clusters)
    intensity_params = {
        "Low":    {"eps": 350, "min_samples": 40, "buffer": 350},
        "Medium": {"eps": 250, "min_samples": 75, "buffer": 250},
        "High":   {"eps": 150, "min_samples": 120, "buffer": 200},
    }
    params = intensity_params[intensity]

    hotspot_areas = None

    # -----------------------------
    # Hotspot computation (DBSCAN)
    # -----------------------------
    if enable_hotspots:
        gdf, hotspot_areas = add_dbscan_hotspots(
            _buildings_gdf=gdf,
            damaged_col="prediction_class",
            damaged_value=1,
            eps_meters=float(params["eps"]),
            min_samples=int(params["min_samples"]),
            buffer_meters=float(params["buffer"])
        )

        # DBSCAN legend
        st.markdown(
            f"""
            <div style="margin-bottom:10px; font-weight:bold;">DBSCAN Hotspot Legend</div>
            <div style="margin-bottom:6px;">
              <span style="font-weight:bold;">Intensity:</span> {intensity}
              &nbsp;&nbsp;|&nbsp;&nbsp;
              <span style="font-weight:bold;">eps:</span> {params["eps"]} m
              &nbsp;&nbsp;|&nbsp;&nbsp;
              <span style="font-weight:bold;">min cluster size:</span> {params["min_samples"]}
            </div>
            <div style="display:flex; gap:20px; align-items:center; margin-bottom: 10px;">
              <div style="width:20px; height:20px; background-color: rgb(0,255,255); border:1px solid white;"></div>
              <span>Undamaged (prediction)</span>
              <div style="width:20px; height:20px; background-color: rgb(255,69,0); border:1px solid white;"></div>
              <span>Damaged (prediction)</span>
              <div style="width:20px; height:20px; background-color: rgba(255,165,0,0.25); border:3px solid orange;"></div>
              <span>DBSCAN hotspot area (dense damaged cluster)</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        tooltip_html = (
            "<b>Building ID:</b> {id}<br>"
            "<b>Actual Label:</b> {label}<br>"
            "<b>Prediction:</b> {prediction_class}<br>"
            "<b>DBSCAN cluster:</b> {db_cluster}<br>"
            "<b>In hotspot?:</b> {db_is_hotspot}"
        )

        # Fill buildings by prediction (same as original)
        def _pred_fill(row):
            try:
                v = int(str(row["prediction_class"]).strip())
            except Exception:
                v = 0
            return [255, 69, 0] if v == 1 else [0, 255, 255]

        gdf["fill_color"] = gdf.apply(_pred_fill, axis=1)
        get_fill_color = "fill_color"

    else:
        # Original legend
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
            "<b>Prediction:</b> {prediction_class}"
        )
        get_fill_color = "prediction_class == '1' || prediction_class == 1 ? [255, 69, 0] : [0, 255, 255]"

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

    # Hotspot overlay layer (orange blobs)
    if enable_hotspots and hotspot_areas is not None and not hotspot_areas.empty:
        hotspot_layer = pdk.Layer(
            "GeoJsonLayer",
            hotspot_areas,
            opacity=0.35,
            stroked=True,
            filled=True,
            get_fill_color=[255, 165, 0, 80],   # orange fill w/ alpha
            get_line_color=[255, 165, 0],       # orange outline
            get_line_width=3,
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
    st.error(f"Failed to load data from BigQuery: {e}")
    st.info("Check your GCP credentials, ensure the BigQuery View exists, and verify dependencies in requirements.txt.")
