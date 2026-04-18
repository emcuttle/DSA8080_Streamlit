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
# Dashboard title
# -----------------------------
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
        SELECT id, label, prediction_class, geometry
        FROM `capstone-project-485905.marshall_v9_seed_75.v_inference_results_geo`
    """
    return client.query(query).to_geodataframe()


# -----------------------------
# GI* Hotspot Function
# -----------------------------
@st.cache_data(show_spinner="Computing cluster hotspots…", ttl=300)
def add_gistar_hotspots(_gdf):

    gdf = _gdf.copy()

    k = 12
    alpha = 0.01
    permutations = 199

    if gdf.crs is None:
        gdf = gdf.set_crs(4326)

    gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())

    pts = gdf_proj.copy()
    pts["geometry"] = pts.geometry.centroid

    y = pd.to_numeric(pts["prediction_class"], errors="coerce").fillna(0).astype(int)
    y = (y == 1).astype(int)

    w = KNN.from_dataframe(pts, k=k)
    w.transform = "R"

    g_local = G_Local(y, w, permutations=permutations, star=True)

    gdf["gi_z"] = g_local.Zs
    gdf["gi_p"] = g_local.p_sim

    reject, pvals_fdr, _, _ = multipletests(
        gdf["gi_p"], alpha=alpha, method="fdr_bh"
    )

    gdf["gi_sig"] = reject

    gdf["gi_cat"] = "Not significant"
    gdf.loc[gdf["gi_sig"] & (gdf["gi_z"] > 0), "gi_cat"] = "Hotspot"
    gdf.loc[gdf["gi_sig"] & (gdf["gi_z"] < 0), "gi_cat"] = "Coldspot"

    return gdf


# -----------------------------
# MAIN APP
# -----------------------------
try:
    gdf = get_bq_data()

    # =========================================================
    # 🔒 CRITICAL FIX: MATCH COLAB EXACTLY
    # =========================================================

    gdf = gdf.sort_values("id").reset_index(drop=True)

    gdf["label"] = gdf["label"].astype(int)
    gdf["prediction_class"] = gdf["prediction_class"].astype(int)

    # Single source of truth (matches Colab merged_df_eroded)
    df_eval = gdf.copy()


    # -----------------------------
    # Model Performance
    # -----------------------------
    with st.expander("View Model Performance Metrics"):
        col1, col2 = st.columns(2)

        with col1:
            st.write("Confusion Matrix")

            y_true = df_eval["label"]
            y_pred = df_eval["prediction_class"]

            cm = confusion_matrix(y_true, y_pred)

            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax,
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
                x=df_eval["prediction_class"],
                ax=ax2,
                palette=["#00FFFF", "#FF4500"],
                order=[0, 1]
            )

            ax2.set_xticklabels(["Undamaged", "Damaged"])
            ax2.set_xlabel("Status")
            ax2.set_ylabel("Count")

            st.pyplot(fig2)


    # -----------------------------
    # HOTSPOTS
    # -----------------------------
    if enable_hotspots:
        df_eval = add_gistar_hotspots(df_eval)

        hotspot_colors = {
            "Hotspot": [255, 0, 0, 200],
            "Coldspot": [0, 0, 255, 200],
            "Not significant": [200, 200, 200, 200],
        }

        df_eval["fill_color"] = df_eval["gi_cat"].map(hotspot_colors)
        df_eval["fill_color"] = df_eval["fill_color"].apply(
            lambda x: x if isinstance(x, list) else [200, 200, 200, 200]
        )

        tooltip_html = """
        <b>ID:</b> {id}<br>
        <b>Actual:</b> {label}<br>
        <b>Prediction:</b> {prediction_class}<br>
        <b>Z-score:</b> {gi_z}<br>
        <b>Category:</b> {gi_cat}
        """

    else:
        df_eval["fill_color"] = np.where(
            df_eval["prediction_class"] == 1,
            [255, 69, 0, 220],
            [0, 255, 255, 220]
        ).tolist()

        tooltip_html = """
        <b>ID:</b> {id}<br>
        <b>Actual:</b> {label}<br>
        <b>Prediction:</b> {prediction_class}
        """


    # -----------------------------
    # MAP (Colab-aligned data)
    # -----------------------------
    df_map = df_eval.copy()

    df_map = df_map.to_crs(4326)

    layer = pdk.Layer(
        "GeoJsonLayer",
        df_map,
        opacity=0.9,
        stroked=False,
        filled=True,
        get_fill_color="fill_color",
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=float(df_map.geometry.centroid.y.mean()),
        longitude=float(df_map.geometry.centroid.x.mean()),
        zoom=14,
        pitch=45,
    )

    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={"html": tooltip_html},
        )
    )


except Exception as e:
    st.error(f"Error: {e}")
