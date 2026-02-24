import streamlit as st
import pandas as pd
import pydeck as pdk
from google.cloud import bigquery
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import geopandas as gpd

from esda import G_Local
from libpysal.weights import KNN, DistanceBand

# Dashboard title
st.title("Marshall CO Wildfire Response: Building Damage Statuses")

# -----------------------------
# 0. Sidebar: Hotspot controls
# -----------------------------
st.sidebar.header("Hotspot Detection (Gi*)")

enable_hotspots = st.sidebar.toggle("Enable hotspot detection", value=False)

neighbor_method = st.sidebar.selectbox(
    "Neighbor method",
    options=["KNN", "DistanceBand"],
    help="KNN = k nearest neighbors; DistanceBand = neighbors within a distance threshold."
)

k = st.sidebar.slider("K (for KNN)", min_value=4, max_value=20, value=10, step=1)
threshold_m = st.sidebar.slider("Distance threshold (meters)", min_value=50, max_value=600, value=200, step=10)

alpha = st.sidebar.select_slider(
    "Significance level (alpha)",
    options=[0.10, 0.05, 0.01],
    value=0.05
)

permutations = st.sidebar.selectbox(
    "Permutations (more = slower, more stable p-values)",
    options=[199, 499, 999],
    index=2
)

# -----------------------------
# 1. BigQuery Connection
# -----------------------------
@st.cache_data
def get_bq_data():
    # Access the credentials from secrets
    if "gcp_service_account" in st.secrets:
        creds_info = st.secrets["gcp_service_account"]
        client = bigquery.Client.from_service_account_info(creds_info)
    else:
        client = bigquery.Client()

    query = """
        SELECT 
            id, 
            label, 
            prediction_class, 
            spatial_geom 
        FROM `capstone-project-485905.marshall_fire_inference.v_marshall_fire_map`
    """
    # BigQuery returns geography; to_geodataframe uses geo stack
    return client.query(query).to_geodataframe()

# -----------------------------
# 1b. Hotspot computation (Gi*)
# -----------------------------
@st.cache_data(show_spinner="Computing Gi* hotspots…")
def add_gistar_hotspots(
    _buildings_gdf: gpd.GeoDataFrame,
    damaged_col: str = "prediction_class",
    damaged_value: int = 1,
    method: str = "KNN",
    k: int = 10,
    threshold_m: float = 200.0,
    permutations: int = 999,
    alpha: float = 0.05
) -> gpd.GeoDataFrame:
    """
    Computes Getis-Ord Gi* (local G) hotspots on building centroids, but returns
    original polygons with added columns:
      - gi_z (z-score)
      - gi_p (permutation p-value)
      - gi_cat (Hotspot / Coldspot / Not significant)
    """
    gdf = _buildings_gdf.copy()

    # Ensure CRS exists (BigQuery sometimes returns None)
    if gdf.crs is None:
        # BigQuery GEOGRAPHY is typically WGS84
        gdf = gdf.set_crs(4326)

    # Use projected CRS for distance calculations (meters)
    if gdf.crs.is_geographic:
        gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())
    else:
        gdf_proj = gdf

    # Use centroids for neighbor calculations
    pts = gdf_proj.copy()
    pts["geometry"] = pts.geometry.centroid

    # Ensure prediction_class is numeric (0/1)
    y = pd.to_numeric(pts[damaged_col], errors="coerce").fillna(0).astype(int).to_numpy()
    y = (y == damaged_value).astype(int)  # damaged=1, undamaged=0

    # Build weights
    if method == "KNN":
        w = KNN.from_dataframe(pts, k=k)
    else:
        w = DistanceBand.from_dataframe(pts, threshold=threshold_m, silence_warnings=True)

    w.transform = "R"  # row standardization

    # Local Gi* (G_Local)
    g_local = G_Local(y, w, permutations=permutations, star=True)

    # Attach results to ORIGINAL gdf (not projected), aligned by row order
    out = gdf.copy()
    out["gi_z"] = g_local.Zs
    out["gi_p"] = g_local.p_sim

    out["gi_cat"] = "Not significant"
    sig = out["gi_p"] < alpha
    out.loc[sig & (out["gi_z"] > 0), "gi_cat"] = "Hotspot (damaged cluster)"
    out.loc[sig & (out["gi_z"] < 0), "gi_cat"] = "Coldspot (undamaged cluster)"

    return out

try:
    gdf = get_bq_data()

    # -----------------------------
    # 2. Model Performance
    # -----------------------------
    with st.expander("📊 View Model Performance Metrics"):
        col1, col2 = st.columns(2)

        with col1:
            st.write("Confusion Matrix")
            y_true = gdf['label'].astype(str).str.strip().astype(int)
            y_pred = gdf['prediction_class'].astype(str).str.strip().astype(int)
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["Undamaged", "Damaged"],
                        yticklabels=["Undamaged", "Damaged"])
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            st.pyplot(fig)

        with col2:
            st.write("Prediction Distribution")
            fig2, ax2 = plt.subplots(figsize=(4, 3))

            sns.countplot(
                x=gdf['prediction_class'].astype(int),
                ax=ax2,
                palette=['#00FFFF', '#FF4500'],
                order=[0, 1]
            )

            ax2.set_xticklabels(["Undamaged", "Damaged"])
            ax2.set_xlabel("Status")
            ax2.set_ylabel("Count")
            st.pyplot(fig2)

    # -----------------------------
    # 3. Apply hotspots (optional)
    # -----------------------------
    if enable_hotspots:
        gdf = add_gistar_hotspots(
            _buildings_gdf=gdf,
            damaged_col="prediction_class",
            damaged_value=1,
            method=neighbor_method,
            k=k,
            threshold_m=float(threshold_m),
            permutations=int(permutations),
            alpha=float(alpha),
        )

        # Hotspot legend + colors
        st.markdown(
            """
            <div style="margin-bottom:10px; font-weight:bold;">Hotspot Legend (Gi*)</div>
            <div style="display:flex; gap:20px; align-items:center; margin-bottom: 10px;">
              <div style="width:20px; height:20px; background-color:#d7191c; border:1px solid white;"></div>
              <span>Hotspot (damaged cluster)</span>
              <div style="width:20px; height:20px; background-color:#2c7bb6; border:1px solid white;"></div>
              <span>Coldspot (undamaged cluster)</span>
              <div style="width:20px; height:20px; background-color:#bdbdbd; border:1px solid white;"></div>
              <span>Not significant</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Assign fill colors as arrays for pydeck
        hotspot_colors = {
            "Hotspot (damaged cluster)": [215, 25, 28],   # red
            "Coldspot (undamaged cluster)": [44, 123, 182], # blue
            "Not significant": [189, 189, 189]            # gray
        }
        gdf["fill_color"] = gdf["gi_cat"].map(hotspot_colors).fillna([189, 189, 189])

        tooltip_html = (
            "<b>Building ID:</b> {id}<br>"
            "<b>Actual Label:</b> {label}<br>"
            "<b>Prediction:</b> {prediction_class}<br>"
            "<b>Gi* z:</b> {gi_z}<br>"
            "<b>p-value:</b> {gi_p}<br>"
            "<b>Category:</b> {gi_cat}"
        )

        get_fill_color = "fill_color"

    else:
        # -----------------------------
        # 3. Original Legend
        # -----------------------------
        st.markdown(f"""
        <div style="display: flex; gap: 20px; align-items: center; margin-bottom: 10px; font-weight: bold;">
          <div style="width: 20px; height: 20px; background-color: rgb(0, 255, 255); border: 1px solid white;"></div>
          <span>Undamaged</span>
          <div style="width: 20px; height: 20px; background-color: rgb(255, 69, 0); border: 1px solid white;"></div>
          <span>Damaged</span>
        </div>
        """, unsafe_allow_html=True)

        tooltip_html = (
            "<b>Building ID:</b> {id}<br>"
            "<b>Actual Label:</b> {label}<br>"
            "<b>Prediction:</b> {prediction_class}"
        )

        # Keep your original deck.gl expression
        get_fill_color = "prediction_class == '1' || prediction_class == 1 ? [255, 69, 0] : [0, 255, 255]"

    # -----------------------------
    # 4. Build PyDeck Map
    # -----------------------------
    polygon_layer = pdk.Layer(
        "GeoJsonLayer",
        gdf,
        opacity=0.9,
        stroked=False,
        filled=True,
        get_fill_color=get_fill_color,
        pickable=True,
    )

    # Safe center: compute in WGS84
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
            layers=[polygon_layer],
            initial_view_state=view_state,
            tooltip={"html": tooltip_html}
        )
    )

except Exception as e:
    st.error(f"Failed to load data from BigQuery: {e}")
    st.info("Check your GCP credentials, ensure the BigQuery View exists, and verify dependencies in requirements.txt.")
