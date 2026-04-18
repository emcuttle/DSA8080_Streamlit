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
# Sidebar Controls
# -----------------------------
st.sidebar.header("Hotspot Detection")

enable_hotspots = st.sidebar.toggle("Show cluster hotspots", value=False)

sensitivity = st.sidebar.select_slider(
    "Hotspot Sensitivity",
    options=["Low", "Medium", "High"],
    value="Medium",
    help="Higher sensitivity detects more hotspots but may include noise."
)

with st.sidebar.expander("What are cluster hotspots?"):
    st.write(
        "Hotspots highlight statistically significant clusters of predicted damaged buildings."
    )


# -----------------------------
# Map Colors (CONSISTENT)
# -----------------------------
COLOR_MAP = {
    "Hotspot": [255, 0, 0],
    "Coldspot": [0, 0, 255],
    "Not significant": [200, 200, 200],
    "Damaged": [255, 69, 0],
    "Undamaged": [0, 255, 255],
}


# -----------------------------
# BigQuery
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
# Gi* Hotspots (KNN ONLY)
# -----------------------------
@st.cache_data(ttl=300)
def add_gistar_hotspots(_gdf, sensitivity):

    gdf = _gdf.copy()

    # map sensitivity → alpha
    alpha_map = {
        "Low": 0.01,
        "Medium": 0.05,
        "High": 0.10
    }
    alpha = alpha_map[sensitivity]

    if gdf.crs is None:
        gdf = gdf.set_crs(4326)

    gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())

    pts = gdf_proj.copy()
    pts["geometry"] = pts.geometry.centroid

    y = (pts["prediction_class"] == 1).astype(int)

    # KNN (stable choice)
    w = KNN.from_dataframe(pts, k=12)
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
# Convex Hull Clusters
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
    # Metrics
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
    # Legend (TOP - FIXED)
    # -----------------------------
    if enable_hotspots:
        st.markdown("### Legend")
        col1, col2, col3 = st.columns(3)

        col1.markdown("🔴 Hotspot")
        col2.markdown("🔵 Coldspot")
        col3.markdown("⚪ Not Significant")

    else:
        st.markdown("### Legend")
        col1, col2 = st.columns(2)

        col1.markdown("🟠 Damaged")
        col2.markdown("🔵 Undamaged")

    # -----------------------------
    # Layers
    # -----------------------------
    layers = []

    if enable_hotspots:

        gdf = add_gistar_hotspots(gdf, sensitivity)

        gdf["fill_color"] = gdf["gi_cat"].map(COLOR_MAP)

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
                    get_fill_color=[255, 0, 0, 60],
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
    # Map
    # -----------------------------
    gdf = gdf.to_crs(4326)

    view_state = pdk.ViewState(
        latitude=gdf.geometry.centroid.y.mean(),
        longitude=gdf.geometry.centroid.x.mean(),
        zoom=14,
    )

    st.pydeck_chart(pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip={"html": tooltip}
    ))

except Exception as e:
    st.error(f"Error: {e}")
