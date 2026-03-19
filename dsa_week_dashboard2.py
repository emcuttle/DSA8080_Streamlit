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
from esda import G_Local
from libpysal.weights import KNN
from statsmodels.stats.multitest import multipletests
from shapely import wkt


# -----------------------------
# Dashboard title
st.title("Marshall CO Wildfire: Building Damage Statuses")
# -----------------------------


# -----------------------------
# Hotspot Toggle
# -----------------------------
st.sidebar.header("Cluster Hotspot Detection")
enable_hotspots = st.sidebar.toggle("Show cluster hotspots", value=False)
with st.sidebar.expander("What are cluster hotspots?"):
    st.write(
        "Cluster hotspots highlight statistically significant clusters of predicted damaged buildings "
        "(Getis-Ord Gi*). This helps identify neighborhoods with concentrated predicted damage."
    )


# -----------------------------
# BigQuery Connection
# -----------------------------
def get_bq_client():
    # access the GCP credentials from secrets.toml
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
    return client.query(query).to_dataframe()


# -----------------------------
# GI* Hotspot Code
# -----------------------------
@st.cache_data(show_spinner="Computing cluster hotspots…", ttl=300)
def add_gistar_hotspots(
    _buildings_gdf: gpd.GeoDataFrame,
    damaged_col: str = "prediction_class",
    damaged_value: int = 1,
    ) -> gpd.GeoDataFrame: 
      """
      Computes Getis-Ord Gi* hotspots on a binary damaged indicator using KNN neighbors.
      Uses FDR (Benjamini–Hochberg) correction to reduce false positives.
      """

      # define parameters
      k = 12  # for mixed spacing
      permutations = 199  # fast enough for a live dashboard
      alpha = 0.01  # more conservative than the standard 0.05 since this dashboard is for search and rescue operations

      gdf = _buildings_gdf.copy()

      # ensure CRS exists or use projected CRS
      if gdf.crs is None:
        gdf = gdf.set_crs(4326)
      if gdf.crs.is_geographic:
          gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())
      else:
          gdf_proj = gdf
  
      # use centroids for KNN calculations
      pts = gdf_proj.copy()
      pts["geometry"] = pts.geometry.centroid
  
      # ensure prediction_class is numeric (0/1)
      y = pd.to_numeric(pts[damaged_col], errors="coerce").fillna(0).astype(int).to_numpy()
      y = (y == damaged_value).astype(int)  # damaged = 1, undamaged = 0
  
      # build weights
      w = KNN.from_dataframe(pts, k=k)
      w.transform = "R" # row standardization
  
      # Local Gi* (G_Local)
      g_local = G_Local(y, w, permutations=permutations, star=True)
  
      # Attach results to original gdf, aligned by row order
      out = gdf.copy()
      out["gi_z"] = g_local.Zs
      out["gi_p"] = g_local.p_sim
  
      # multiple-testing correction (FDR / BH) - protecting against false positives
      pvals = out["gi_p"].fillna(1.0).to_numpy()
      reject, pvals_fdr, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
      out["gi_p_fdr"] = pvals_fdr
      out["gi_sig"] = reject
  
      # categorize hotspot using FDR-adjusted significance
      out["gi_cat"] = "Not significant"
      sig = out["gi_sig"]
      out.loc[sig & (out["gi_z"] > 0), "gi_cat"] = "Hotspot (damaged cluster)"
      out.loc[sig & (out["gi_z"] < 0), "gi_cat"] = "Coldspot (undamaged cluster)"

      return out

try:
    # gdf = get_bq_data()
    df = get_bq_data()

    # --- NEW REPROJECTION CODE START ---
    # 1. Convert WKT string to actual geometry objects
    df['geometry'] = df['geometry'].apply(wkt.loads)
    
    # 2. Convert DataFrame to GeoDataFrame and set initial CRS (UTM 13N)
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:32613")
    
    # 3. Transform to WGS84 (Degrees) so the maps and stats work
    gdf = gdf.to_crs(epsg=4326)
    # --- NEW REPROJECTION CODE END ---

    # -----------------------------
    # Model Performance
    # -----------------------------
    with st.expander("View Model Performance Metrics"):
        # ... rest of your code ...
  

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
    # Enabling cluserting hotspots
    # -----------------------------
    if enable_hotspots:
        gdf = add_gistar_hotspots(
            _buildings_gdf=gdf,
            damaged_col="prediction_class",
            damaged_value=1,
        )

        # hotspot legend
        st.markdown(
            """
            <div style="margin-bottom:10px; font-weight:bold;">Hotspot Legend</div>
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

        # assign fill colors as arrays for pydeck
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
            "<b>p-value (raw):</b> {gi_p}<br>"
            "<b>p-value (FDR):</b> {gi_p_fdr}<br>"
            "<b>Category:</b> {gi_cat}"
        )
        get_fill_color = "fill_color"

    else:
        # -----------------------------
        # original Legend
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
