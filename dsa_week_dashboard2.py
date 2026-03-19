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
import shapely.ops as ops
from functools import partial
import pyproj

# -----------------------------
# Dashboard title
# -----------------------------
st.set_page_config(layout="wide")
st.title("Marshall CO Wildfire: Building Damage Statuses")

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
@st.cache_data(show_spinner="Computing cluster hotspots...", ttl=300)
def add_gistar_hotspots(
    _buildings_gdf: gpd.GeoDataFrame,
    damaged_col: str = "prediction_class",
    damaged_value: int = 1,
    ) -> gpd.GeoDataFrame: 

    k = 12
    permutations = 199
    alpha = 0.01

    gdf = _buildings_gdf.copy()

    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    
    if gdf.crs.is_geographic:
        gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())
    else:
        gdf_proj = gdf

    pts = gdf_proj.copy()
    pts["geometry"] = pts.geometry.centroid

    y = pd.to_numeric(pts[damaged_col], errors="coerce").fillna(0).astype(int).to_numpy()
    y = (y == damaged_value).astype(int)

    w = KNN.from_dataframe(pts, k=k)
    w.transform = "R"

    g_local = G_Local(y, w, permutations=permutations, star=True)

    out = gdf.copy()
    out["gi_z"] = g_local.Zs
    out["gi_p"] = g_local.p_sim

    pvals = out["gi_p"].fillna(1.0).to_numpy()
    reject, pvals_fdr, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
    out["gi_p_fdr"] = pvals_fdr
    out["gi_sig"] = reject

    out["gi_cat"] = "Not significant"
    sig = out["gi_sig"]
    out.loc[sig & (out["gi_z"] > 0), "gi_cat"] = "Hotspot (damaged cluster)"
    out.loc[sig & (out["gi_z"] < 0), "gi_cat"] = "Coldspot (undamaged cluster)"

    return out

# -----------------------------
# Data Loading and Transformation
# -----------------------------
try:
    df = get_bq_data()
    # Convert WKT to geometry
    df['geometry'] = df['geometry'].apply(wkt.loads)
    # Create GeoDataFrame from UTM 13N
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:32613")
    # Transform to Degrees for mapping
    gdf = gdf.to_crs(epsg=4326)

    # -----------------------------
    # Model Performance Expander
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
            # Fixed Seaborn warning by assigning hue
            sns.countplot(
                x=gdf['prediction_class'].astype(int),
                hue=gdf['prediction_class'].astype(int),
                ax=ax2,
                palette=['#00FFFF', '#FF4500'],
                order=[0, 1],
                legend=False
            )
            ax2.set_xticks([0, 1])
            ax2.set_xticklabels(["Undamaged", "Damaged"])
            ax2.set_xlabel("Status")
            ax2.set_ylabel("Count")

# except Exception as e:
#     st.error(f"Failed to load data from BigQuery: {e}")
#     st.info("Check your GCP credentials, ensure the BigQuery View exists, and verify dependencies in requirements.txt.")
