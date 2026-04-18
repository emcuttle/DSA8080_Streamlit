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


# -----------------------------
# Title
# -----------------------------
st.title("Marshall CO Wildfire: Building Damage Statuses")


# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("Cluster Hotspot Detection")

enable_hotspots = st.sidebar.toggle("Show cluster hotspots", value=False)

sensitivity = st.sidebar.select_slider(
    "Hotspot Sensitivity",
    options=["Low", "Medium", "High"],
    value="Medium"
)

with st.sidebar.expander("What are cluster hotspots?"):
    st.write(
        "Hotspots are statistically significant clusters of predicted damage using Getis-Ord Gi*."
    )


# -----------------------------
# BigQuery
# -----------------------------
def get_bq_client():
    if "gcp_service_account" in st.secrets:
        return bigquery.Client.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
    return bigquery.Client()


@st.cache_data(ttl=0)
def get_bq_data():
    client = get_bq_client()
    query = """
        SELECT id, label, prediction_class, geometry, run_timestamp
        FROM `capstone-project-485905.capstone_model_results.v_latest_marshall`
    """
    return client.query(query).to_geodataframe()


# -----------------------------
# GI* HOTSPOT FUNCTION (SIMPLIFIED + STABLE)
# -----------------------------
@st.cache_data(show_spinner="Computing hotspots…", ttl=0)
def add_gistar_hotspots(_gdf, sensitivity):

    gdf = _gdf.copy()

    # FIXED alpha (prevents instability)
    alpha = 0.05

    # ONLY sensitivity driver = k
    k_map = {
        "Low": 8,
        "Medium": 12,
        "High": 18
    }
    k = k_map[sensitivity]

    if gdf.crs is None:
        gdf = gdf.set_crs(4326)

    gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())

    pts = gdf_proj.copy()
    pts["geometry"] = pts.geometry.centroid

    y = pd.to_numeric(pts["prediction_class"], errors="coerce").fillna(0).astype(int)
    y = (y == 1).astype(int)

    w = KNN.from_dataframe(pts, k=k)
    w.transform = "R"

    g_local = G_Local(y, w, permutations=199, star=True)

    gdf["gi_z"] = g_local.Zs
    gdf["gi_p"] = g_local.p_sim

    reject, _, _, _ = multipletests(gdf["gi_p"], alpha=alpha, method="fdr_bh")

    gdf["gi_cat"] = "Not significant"
    gdf.loc[reject & (gdf["gi_z"] > 0), "gi_cat"] = "Hotspot"
    gdf.loc[reject & (gdf["gi_z"] < 0), "gi_cat"] = "Coldspot"

    return gdf


# -----------------------------
# MAIN
# -----------------------------
try:
    gdf = get_bq_data()

    # -----------------------------
    # METRICS
    # -----------------------------
    with st.expander("View Model Performance Metrics"):
        col1, col2 = st.columns(2)

        with col1:
            cm = confusion_matrix(
                gdf["label"].astype(int),
                gdf["prediction_class"].astype(int)
            )
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

        with col2:
            fig2, ax2 = plt.subplots()
            sns.countplot(x=gdf["prediction_class"].astype(int), ax=ax2)
            st.pyplot(fig2)

    # -----------------------------
    # LEGEND (UNCHANGED — EXACT VERSION YOU WANTED)
    # -----------------------------
    if enable_hotspots:
        st.markdown(
            """
            <div style="display: flex; gap: 20px; align-items: center; margin-bottom: 10px; font-weight: bold;">
              <div style="width: 20px; height: 20px; background-color: rgb(255, 0, 0); border: 1px solid white;"></div>
              <span>Hotspot</span>
              <div style="width: 20px; height: 20px; background-color: rgb(0, 0, 255); border: 1px solid white;"></div>
              <span>Coldspot</span>
              <div style="width: 20px; height: 20px; background-color: rgb(200, 200, 200); border: 1px solid white;"></div>
              <span>Not significant</span>
            </div>
            """,
            unsafe_allow_html=True
        )
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

    # -----------------------------
    # MAP
    # -----------------------------
    layers = []

    if enable_hotspots:

        gdf = add_gistar_hotspots(gdf, sensitivity)

        color_map = {
            "Hotspot": [255, 0, 0],
            "Coldspot": [0, 0, 255],
            "Not significant": [200, 200, 200]
        }

        gdf["fill_color"] = gdf["gi_cat"].map(color_map)

        layers.append(
            pdk.Layer(
                "GeoJsonLayer",
                gdf,
                get_fill_color="fill_color",
                pickable=True,
            )
        )

        tooltip = "<b>ID:</b> {id}<br><b>Z:</b> {gi_z}"

    else:
        layers.append(
            pdk.Layer(
                "GeoJsonLayer",
                gdf,
                get_fill_color="prediction_class == 1 ? [255,69,0] : [0,255,255]",
                pickable=True,
            )
        )

        tooltip = "<b>ID:</b> {id}<br><b>Prediction:</b> {prediction_class}"

    # -----------------------------
    # VIEW
    # -----------------------------
    gdf = gdf.to_crs(4326)

    view_state = pdk.ViewState(
        latitude=gdf.geometry.centroid.y.mean(),
        longitude=gdf.geometry.centroid.x.mean(),
        zoom=14,
    )

    st.pydeck_chart(
        pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip={"html": tooltip}
        )
    )

except Exception as e:
    st.error(f"Error: {e}")
