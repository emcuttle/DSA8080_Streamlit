import streamlit as st
import pandas as pd
import pydeck as pdk
from google.cloud import bigquery
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Dashboard title
st.title("Marshall CO Wildfire Response: Building Damage Statuses")

# -----------------------------
# 1. BigQuery Connection
# -----------------------------
# This function connects to BigQuery and pulls the spatial data
@st.cache_data
def get_bq_data():
    # 1. Access the credentials from the secrets.toml file
    if "gcp_service_account" in st.secrets:
        creds_info = st.secrets["gcp_service_account"]
        client = bigquery.Client.from_service_account_info(creds_info)
    else:
        client = bigquery.Client()

    # 2. The query
    query = """
        SELECT 
            id, 
            label, 
            prediction_class, 
            spatial_geom 
        FROM `capstone-project-485905.marshall_fire_inference.v_marshall_fire_map`
    """
    return client.query(query).to_geodataframe()


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
    # 3. Legend
    # -----------------------------
    st.markdown(f"""
    <div style="display: flex; gap: 20px; align-items: center; margin-bottom: 10px; font-weight: bold;">
      <div style="width: 20px; height: 20px; background-color: rgb(0, 255, 255); border: 1px solid white;"></div>
      <span>Undamaged</span>
      <div style="width: 20px; height: 20px; background-color: rgb(255, 69, 0); border: 1px solid white;"></div>
      <span>Damaged</span>
    </div>
    """, unsafe_allow_html=True)

    # -----------------------------
    # 4. Build PyDeck Map
    # -----------------------------
    polygon_layer = pdk.Layer(
        "GeoJsonLayer",
        gdf,
        opacity=0.9,
        stroked=False,
        filled=True,
        get_fill_color="prediction_class == '1' || prediction_class == 1 ? [255, 69, 0] : [0, 255, 255]",
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=gdf.geometry.centroid.y.mean(),
        longitude=gdf.geometry.centroid.x.mean(),
        zoom=14,
        pitch=45,
    )

    st.pydeck_chart(
        pdk.Deck(
            layers=[polygon_layer],
            initial_view_state=view_state,
            tooltip={
                "html": "<b>Building ID:</b> {building_id}<br>"
                        "<b>Actual Label:</b> {label}<br>"
                        "<b>Prediction:</b> {prediction_class}"
            }
        )
    )

except Exception as e:
    st.error(f"Failed to load data from BigQuery: {e}")
    st.info("Check your GCP credentials and ensure the BigQuery View exists.")