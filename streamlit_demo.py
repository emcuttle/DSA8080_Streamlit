import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely import wkt
import pydeck as pdk
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# Load your CSV
# -----------------------------
st.title("Marhsall Wildfire Building Damage Map")

df = pd.read_csv("marshall_fire_inference.csv")


# -----------------------------
# Display Confusion Matrix
# -----------------------------
st.subheader("Model Performance: Confusion Matrix")

# Compute confusion matrix
cm = confusion_matrix(df['label'], df['prediction_class'])

# Plot using seaborn
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Undamaged","Damaged"], yticklabels=["Undamaged","Damaged"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# prediction_counts = df['prediction_class'].value_counts()
# label_counts = df['label'].value_counts()
# print(prediction_counts)
# print(label_counts)

# Convert WKT polygons to geometry
df["geometry"] = df["geometry"].apply(wkt.loads)

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry="geometry")

# IMPORTANT: set the CRS of your current coordinates
# Your numbers look like UTM zone 10N (common for CA wildfire datasets)
# If different, adjust this to your correct EPSG.
gdf = gdf.set_crs(epsg=32613)

# Convert to WGS84 lat/lon for web mapping
gdf = gdf.to_crs(epsg=4326)

# Extract polygon coordinate arrays for pydeck
def polygon_to_coordinates(geom):
    # pydeck expects [[[lon, lat], ...]] format
    return [[list(coord) for coord in geom.exterior.coords]]

gdf["coords"] = gdf.geometry.apply(polygon_to_coordinates)

# -----------------------------
# Build PyDeck polygon layer
# -----------------------------
polygon_layer = pdk.Layer(
    "PolygonLayer",
    gdf,
    get_polygon="coords",
    get_fill_color="[255 * prediction_class, 50, 120]",
    get_line_color=[0, 0, 0],
    line_width_min_pixels=1,
    pickable=True,
    auto_highlight=True,
)

# Center map on your data
mid_lat = gdf.geometry.centroid.y.mean()
mid_lon = gdf.geometry.centroid.x.mean()

view_state = pdk.ViewState(
    latitude=mid_lat,
    longitude=mid_lon,
    zoom=14,
    pitch=45,
)

# -----------------------------
# Add legend
# -----------------------------
st.markdown("### Legend")
st.markdown("""
<div style="display: flex; gap: 20px; align-items: center;">
  <div style="width: 20px; height: 20px; background-color: rgb(0,50,120);"></div>
  <span>Undamaged (prediction_class = 0)</span>
  <div style="width: 20px; height: 20px; background-color: rgb(255,50,120);"></div>
  <span>Damaged (prediction_class = 1)</span>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Render Streamlit map
# -----------------------------
st.pydeck_chart(
    pdk.Deck(
        layers=[polygon_layer],
        initial_view_state=view_state,
        tooltip={
            "html": "<b>ID:</b> {id}<br>"
                    "<b>Label:</b> {label}<br>"
                    "<b>Prediction:</b> {prediction_class}"
        }
    )
)