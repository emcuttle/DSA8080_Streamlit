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
st.title("Marshall CO Wildfire: Building Damage Statuses")

# -----------------------------
# Sidebar: Hotspot controls
# -----------------------------
st.sidebar.header("Hotspot Detection")

enable_hotspots = st.sidebar.toggle("Enable hotspot detection", value=False)

neighbor_method = st.sidebar.selectbox(
    "Neighbor method",
    options=["KNN", "DistanceBand"],
    help="KNN = K-Nearest Neighbor Algorithm; DistanceBand = Neighbors within a fixed distance."
)

k = st.sidebar.slider("K (for KNN)", min_value=4, max_value=20, value=10, step=1)
threshold_m = st.sidebar.slider("Distance threshold (meters)", min_value=2, max_value=50, value=20, step=2)

alpha = st.sidebar.select_slider(
    "Significance level (alpha)",
    options=[0.01, 0.05, 0.10],
    value=0.05
)

permutations = st.sidebar.selectbox(
    "Permutations (more = slower, more stable p-values)",
    options=[199, 499, 999],
    index=2
)

# -----------------------------
# BigQuery Connection
# -----------------------------
# @st.cache_data
def get_bq_client():
    # Access the GCP credentials from secrets.toml
    if "gcp_service_account" in st.secrets:
        creds_info = st.secrets["gcp_service_account"]
        return bigquery.Client.from_service_account_info(creds_info)
    return bigquery.Client()
    #     client = bigquery.Client.from_service_account_info(creds_info)
    # else:
    #     client = bigquery.Client()

# pulling view data from BQ
@st.cache_data
def get_bq_data():
    client = get_bq_client()
    query = """
        SELECT 
            id, 
            label, 
            prediction_class, 
            spatial_geom
        FROM `capstone-project-485905.marshall_v9_seed_75.v_inference_results_geo`
    """
    return client.query(query).to_geodataframe()

# creating KPI function to calculate total area affected (based on total area of buildings classified as damaged)
@st.cache_data(ttl=300)
def get_bq_kpis():
    client = get_bq_client()
    kpi_query = """
        SELECT
            COUNT(*) AS total_buildings,
            SUM(CASE WHEN prediction_class = 1 THEN 1 ELSE 0 END) AS predicted_damaged_buildings,
            SUM(CASE WHEN prediction_class = 1 THEN ST_AREA(spatial_geom) ELSE 0 END) / 1e6 AS predicted_damaged_area_km2,
            SUM(CASE WHEN prediction_class = 1 THEN ST_AREA(spatial_geom) ELSE 0 END) * 0.000247105 AS predicted_damaged_area_acres
        FROM `capstone-project-485905.marshall_v9_seed_75.v_inference_results_geo`
    """
    return client.query(kpi_query).to_dataframe()

# display KPI cards
kpi_df = get_bq_kpis()

total_buildings = int(kpi_df.loc[0, "total_buildings"] or 0)
pred_damaged = int(kpi_df.loc[0, "predicted_damaged_buildings"] or 0)
area_km2 = float(kpi_df.loc[0, "predicted_damaged_area_km2"] or 0.0)
area_acres = float(kpi_df.loc[0, "predicted_damaged_area_acres"] or 0.0)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Buildings", f"{total_buildings:,}")
c2.metric("Predicted Damaged (count)", f"{pred_damaged:,}")
c3.metric("Predicted Damaged Area", f"{area_km2:,.2f} km²")
c4.metric("Predicted Damaged Area", f"{area_acres:,.0f} acres")
    

# -----------------------------
# GI* Hotspot Code
# -----------------------------
@st.cache_data(show_spinner="Computing hotspots…")
def add_gistar_hotspots(
    _buildings_gdf: gpd.GeoDataFrame,
    damaged_col: str = "prediction_class",
    damaged_value: int = 1,
    method: str = "KNN",
    k: int = 10,
    threshold_m: float = 50.0,
    permutations: int = 999,
    alpha: float = 0.05
) -> gpd.GeoDataFrame:
    gdf = _buildings_gdf.copy()

    if gdf.crs is None:
        gdf = gdf.set_crs(4326)

    # Use projected CRS for distance calculations (meters)
    if gdf.crs.is_geographic:
        gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())
    else:
        gdf_proj = gdf

    # Use centroids for KNN calculations
    pts = gdf_proj.copy()
    pts["geometry"] = pts.geometry.centroid

    # Ensure prediction_class is numeric (0/1)
    y = pd.to_numeric(pts[damaged_col], errors="coerce").fillna(0).astype(int).to_numpy()
    y = (y == damaged_value).astype(int)  # damaged = 1, undamaged = 0

    # Build weights
    if method == "KNN":
        w = KNN.from_dataframe(pts, k=k)
    else:
        w = DistanceBand.from_dataframe(pts, threshold=threshold_m, silence_warnings=True)

    w.transform = "R"  # row standardization

    # Local Gi* (G_Local)
    g_local = G_Local(y, w, permutations=permutations, star=True)

    # Attach results to original gdf, aligned by row order
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
    # Model Performance
    # -----------------------------
    with st.expander("View Model Performance Metrics"):
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
    # Apply hotspots
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

        # Hotspot legend
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
        gdf["fill_color"] = gdf["gi_cat"].map(hotspot_colors)
        
        default_color = [189, 189, 189]
        gdf["fill_color"] = gdf["fill_color"].apply(
            lambda x: x if isinstance(x, list) else default_color
        )


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
        # Original Legend
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

        get_fill_color = "prediction_class == '1' || prediction_class == 1 ? [255, 69, 0] : [0, 255, 255]"

    # -----------------------------
    # Build Map
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
