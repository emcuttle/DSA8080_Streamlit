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
# Sidebar Toggle
# -----------------------------
st.sidebar.header("Cluster Hotspot Detection")
enable_hotspots = st.sidebar.toggle("Show cluster hotspots", value=False)


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
        SELECT id, label, prediction_class, geometry
        FROM `capstone-project-485905.marshall_v9_seed_75.v_inference_results_geo`
    """
    return client.query(query).to_geodataframe()


# -----------------------------
# GI* FUNCTION
# -----------------------------
@st.cache_data(show_spinner="Computing cluster hotspots…", ttl=300)
def add_gistar_hotspots(gdf):

    k = 12
    gdf = gdf.copy()

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

    reject, pvals_fdr, _, _ = multipletests(gdf["gi_p"], alpha=0.01, method="fdr_bh")
    gdf["gi_sig"] = reject

    gdf["gi_cat"] = "Not significant"
    gdf.loc[gdf["gi_sig"] & (gdf["gi_z"] > 0), "gi_cat"] = "Hotspot"
    gdf.loc[gdf["gi_sig"] & (gdf["gi_z"] < 0), "gi_cat"] = "Coldspot"

    # -----------------------------
    # INTENSITY
    # -----------------------------
    gdf["intensity"] = "None"
    gdf.loc[gdf["gi_z"] >= 2.58, "intensity"] = "High"
    gdf.loc[(gdf["gi_z"] >= 1.96) & (gdf["gi_z"] < 2.58), "intensity"] = "Medium"
    gdf.loc[(gdf["gi_z"] >= 1.65) & (gdf["gi_z"] < 1.96), "intensity"] = "Low"

    return gdf


# -----------------------------
# CREATE CONVEX HULLS
# -----------------------------
def create_hotspot_hulls(gdf):

    hotspots = gdf[gdf["gi_cat"] == "Hotspot"].copy()

    if hotspots.empty:
        return None

    # project for clustering
    hotspots_proj = hotspots.to_crs(hotspots.estimate_utm_crs())

    coords = np.array(list(zip(
        hotspots_proj.geometry.centroid.x,
        hotspots_proj.geometry.centroid.y
    )))

    # DBSCAN just for grouping nearby hotspots
    clustering = DBSCAN(eps=100, min_samples=5).fit(coords)
    hotspots_proj["cluster"] = clustering.labels_

    hulls = []

    for cluster_id in set(clustering.labels_):
        if cluster_id == -1:
            continue

        cluster_points = hotspots_proj[hotspots_proj["cluster"] == cluster_id]

        if len(cluster_points) < 3:
            continue

        hull = cluster_points.unary_union.convex_hull
        hulls.append(hull)

    if not hulls:
        return None

    hulls_gdf = gpd.GeoDataFrame(geometry=hulls, crs=hotspots_proj.crs)
    return hulls_gdf.to_crs(4326)


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
            y_true = gdf['label'].astype(int)
            y_pred = gdf['prediction_class'].astype(int)
            cm = confusion_matrix(y_true, y_pred)

            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

        with col2:
            fig2, ax2 = plt.subplots()
            sns.countplot(x=gdf['prediction_class'].astype(int), ax=ax2)
            st.pyplot(fig2)

    # -----------------------------
    # HOTSPOTS
    # -----------------------------
    layers = []

    if enable_hotspots:

        gdf = add_gistar_hotspots(gdf)

        # intensity bars
        st.subheader("Hotspot Intensity Distribution")
        intensity_counts = (
            gdf[gdf["gi_cat"] == "Hotspot"]["intensity"]
            .value_counts()
            .reindex(["High", "Medium", "Low"], fill_value=0)
        )

        fig3, ax3 = plt.subplots()
        intensity_counts.plot(kind="bar", ax=ax3)
        st.pyplot(fig3)

        # colors
        colors = {
            "Hotspot": [255, 0, 0],
            "Coldspot": [0, 0, 255],
            "Not significant": [200, 200, 200]
        }
        gdf["fill_color"] = gdf["gi_cat"].map(colors)

        # base layer
        layers.append(
            pdk.Layer(
                "GeoJsonLayer",
                gdf,
                get_fill_color="fill_color",
                pickable=True,
            )
        )

        # -----------------------------
        # ADD CONVEX HULL LAYER
        # -----------------------------
        hulls_gdf = create_hotspot_hulls(gdf)

        if hulls_gdf is not None:
            layers.append(
                pdk.Layer(
                    "GeoJsonLayer",
                    hulls_gdf,
                    get_fill_color=[255, 0, 0, 80],  # transparent red
                    get_line_color=[255, 0, 0],
                    line_width_min_pixels=2,
                    pickable=False,
                )
            )

        tooltip_html = "<b>ID:</b> {id}<br><b>Z:</b> {gi_z}<br><b>Intensity:</b> {intensity}"

    else:
        layers.append(
            pdk.Layer(
                "GeoJsonLayer",
                gdf,
                get_fill_color="prediction_class == 1 ? [255,69,0] : [0,255,255]",
                pickable=True,
            )
        )
        tooltip_html = "<b>ID:</b> {id}<br><b>Prediction:</b> {prediction_class}"

    # -----------------------------
    # MAP VIEW
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
        tooltip={"html": tooltip_html}
    ))

except Exception as e:
    st.error(f"Error: {e}")
