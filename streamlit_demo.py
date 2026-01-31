import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely import wkt
import pydeck as pdk
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. CONFIGURATION
# -----------------------------
st.set_page_config(layout="wide")
st.title("Marshall CO Wildfire Response: Building Damage Statuses")

DATA_URL = "https://storage.googleapis.com/sandbox-marshall-fire-model-outputs/marshall_fire_model_output.csv"

# -----------------------------
# 2. DATA LOADING
# -----------------------------
@st.cache_data
def load_data(url):
    try:
        # Read directly from the URL
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"Error loading data from Cloud: {e}")
        return pd.DataFrame()

# Load the data
df = load_data(DATA_URL)

if not df.empty:
    # -----------------------------
    # 3. DATA PROCESSING
    # -----------------------------
    # Convert WKT polygons to geometry
    df["geometry"] = df["geometry"].apply(wkt.loads)

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry="geometry")

    # Set CRS and convert to Web Mercator (Lat/Lon)
    # Ensure your original CSV WKT is actually in EPSG:32613
    gdf = gdf.set_crs(epsg=32613) 
    gdf = gdf.to_crs(epsg=4326)

    # Extract polygon coordinate arrays for pydeck
    def polygon_to_coordinates(geom):
        return [[list(coord) for coord in geom.exterior.coords]]

    gdf["coords"] = gdf.geometry.apply(polygon_to_coordinates)

    # -----------------------------
    # 4. VISUALIZATION (PyDeck)
    # -----------------------------
    st.subheader("Live Damage Assessment Map")
    
    # Define the PyDeck Layer
    polygon_layer = pdk.Layer(
        "PolygonLayer",
        gdf,
        get_polygon="coords",
        get_fill_color="prediction_class == 1 ? [255, 50, 120] : [50, 150, 255]",
        get_line_color=[0, 0, 0],
        line_width_min_pixels=1,
        pickable=True,
        auto_highlight=True,
        opacity=0.8
    )

    # Calculate View State
    mid_lat = gdf.geometry.centroid.y.mean()
    mid_lon = gdf.geometry.centroid.x.mean()

    view_state = pdk.ViewState(
        latitude=mid_lat,
        longitude=mid_lon,
        zoom=14,
        pitch=45,
    )

    # Render Map
    st.pydeck_chart(
        pdk.Deck(
            layers=[polygon_layer],
            initial_view_state=view_state,
            tooltip={
                "html": "<b>ID:</b> {id}<br><b>Prediction:</b> {prediction_class}"
            }
        )
    )

    # -----------------------------
    # 5. LEGEND & METRICS
    # -----------------------------
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Legend")
        st.markdown("""
        <div style="display: flex; flex-direction: column; gap: 10px;">
          <div style="display: flex; align-items: center; gap: 10px;">
            <div style="width: 20px; height: 20px; background-color: rgb(50,150,255);"></div>
            <span>Undamaged</span>
          </div>
          <div style="display: flex; align-items: center; gap: 10px;">
            <div style="width: 20px; height: 20px; background-color: rgb(255,50,120);"></div>
            <span>Damaged</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("### Statistics")
        pred_counts = df['prediction_class'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 2))
        sns.barplot(x=pred_counts.index, y=pred_counts.values, ax=ax, palette=["#3296FF", "#FF3278"])
        ax.set_xticklabels(["Undamaged", "Damaged"])
        st.pyplot(fig)
