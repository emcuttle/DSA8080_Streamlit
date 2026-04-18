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
from sklearn.cluster import DBSCAN


# -----------------------------
# Dashboard title
# -----------------------------
st.title("Marshall CO Wildfire: Building Damage Statuses")


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Cluster Hotspot Detection")

enable_hotspots = st.sidebar.toggle("Show cluster hotspots", value=False)

sensitivity = st.sidebar.select_slider(
    "Hotspot Sensitivity",
    options=["Low", "Medium", "High"],
    value="Medium",
    help="Controls how strict hotspot detection is."
)


# -----------------------------
# BigQuery Connection
# -----------------------------
def get_bq_client():
    if "gcp_service_account" in st.secrets:
        return bigquery.Client.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
    return bigquery.Client()


@st.cache_data(ttl=300)
def get_bq_data():
    client = get_bq_client()
    query = """
        SELECT id, label, prediction_class, geometry
        FROM `capstone-project-485905.marshall_v9_seed_75.v_inference_results_geo`
    """
    return client.query(query).to_geodataframe()


# -----------------------------
# GI* HOTSPOT FUNCTION (FIXED SIGNIFICANCE)
# -----------------------------
@st.cache_data(show_spinner="Computing cluster hotspots…", ttl=300)
def add_gistar_hotspots(_gdf, sensitivity):

    gdf = _gdf.copy()

    # Map sensitivity → alpha (FIXED LOGIC)
    alpha_map = {
        "Low": 0.10,     # more hotspots
        "Medium": 0.05,
        "High": 0.01     # stricter
    }
    alpha = alpha_map[sensitivity]

    if gdf.crs is None:
        gdf = gdf.set_crs(4326)

    gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())

    pts = gdf_proj.copy()
    pts["geometry"] = pts.geometry.centroid

    y = (pts["prediction_class"] == 1).astype(int)

    w = KNN.from_dataframe(pts, k=12)
    w.transform = "R"

    g_local = G_Local(y, w, permutations=199, star=True)

    gdf["gi_z"] = g_local.Zs
    gdf["gi_p"] = g_local.p_sim

    # IMPORTANT FIX: less aggressive collapse of signal
    reject, _, _, _ = multipletests(gdf["gi_p"], alpha=alpha, method="fdr_bh")

    gdf["gi_cat"] = "Not significant"
    gdf.loc[reject & (gdf["gi_z"] > 0), "gi_cat"] = "Hotspot"
    gdf.loc[reject & (gdf["gi_z"] < 0), "gi_cat"] = "Coldspot"

    return gdf


# -----------------------------
# CONVEX HULLS (UNCHANGED)
# -----------------------------
def create_hotspot_hulls(gdf):

    hotspots = gdf[gdf["gi_cat"] == "Hotspot"].copy()

    if hotspots.empty:
        return None

    hotspots_proj = hotspots.to_crs(hotspots.estimate_utm_crs())

    coords = np.array(list(zip(
        hotspots_proj.geometry.centroid.x,
        hotspots_proj.geometry.centroid.y
    )))

    clustering = DBSCAN(eps=120, min_samples=4).fit(coords)
    hotspots_proj["cluster"] = clustering.labels_

    hulls = []

    for cid in set(clustering.labels_):
        if cid == -1:
            continue

        cluster_pts = hotspots_proj[hotspots_proj["cluster"] == cid]

        if len(cluster_pts) < 3:
            continue

        hulls.append(cluster_pts.unary_union.convex_hull)

    if not hulls:
        return None

    return gpd.GeoDataFrame(geometry=hulls, crs=hotspots_proj.crs).to_crs(4326)


# -----------------------------
# MAIN
# -----------------------------
try:
    gdf = get_bq_data()

    # -----------------------------
    # PERFORMANCE METRICS
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
    # LEGEND (RESTORED EXACT STYLE)
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
    # LAYERS
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

        hulls = create_hotspot_hulls(gdf)

        if hulls is not None:
            layers.append(
                pdk.Layer(
                    "GeoJsonLayer",
                    hulls,
                    get_fill_color=[255, 0, 0, 70],
                    get_line_color=[255, 0, 0],
                    line_width_min_pixels=2,
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
    # MAP
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
