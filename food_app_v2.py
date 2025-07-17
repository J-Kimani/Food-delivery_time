import streamlit as st
import numpy as np
import pandas as pd
import pydeck as pdk
from keras.models import load_model
from math import radians, sin, cos, sqrt, atan2

st.set_page_config(page_title="Food Delivery ETA", page_icon="🍔", layout="centered")

# ------------------------------
# wallpaper
# ------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1600891964599-f61ba0e24092");
        background-attachment: fixed;
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
    }
    .block-container {
        background-color: rgba(0, 0, 0, 0.78);
        padding: 2rem;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ------------------------------
# Load the trained model
# ------------------------------
@st.cache_resource
def load_trained_model():
    model = load_model("food_delivery_model.h5", compile=False)
    return model

model = load_trained_model()

# ------------------------------
# Function to calculate Haversine distance
# ------------------------------
def calc_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# ------------------------------
# UI Layout
# ------------------------------

st.title("Food Delivery Time Prediction")

st.markdown("Estimate the time it will take for food to be delivered based on delivery partner details and location.")

# --- Input Section ---
st.header("User Input")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Delivery Partner Age", min_value=18, max_value=65, value=30)
    rest_lat = st.number_input("Restaurant Latitude", format="%.6f", value=12.9715987)
    deliv_lat = st.number_input("Delivery Latitude", format="%.6f", value=12.2958104)

with col2:
    rating = st.slider("Partner Rating", 1.0, 5.0, value=4.0, step=0.1)
    rest_lon = st.number_input("Restaurant Longitude", format="%.6f", value=77.594566)
    deliv_lon = st.number_input("Delivery Longitude", format="%.6f", value=76.6393805)

# --- Predict Button ---
if st.button("📦 Predict Delivery Time"):
    with st.spinner("⏳ Calculating..."):

        # Calculate distance
        distance = calc_distance(rest_lat, rest_lon, deliv_lat, deliv_lon)

        # Format input and predict
        input_data = np.array([[age, rating, distance]])
        input_reshaped = input_data.reshape((1, 3, 1))
        prediction = model.predict(input_reshaped, verbose=0)[0][0]

    # Show prediction results
    st.success(f"🕒 Estimated Delivery Time: **{prediction:.2f} minutes**")
    st.info(f"📏 Distance: **{distance:.2f} km**")
    st.balloons()

    # --- Route Map ---
    st.subheader("🗺️ Delivery Route Map")
    map_data = pd.DataFrame({
        'lat': [rest_lat, deliv_lat],
        'lon': [rest_lon, deliv_lon],
        'label': ['Restaurant', 'Delivery Location']
    })

    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(
            latitude=(rest_lat + deliv_lat) / 2,
            longitude=(rest_lon + deliv_lon) / 2,
            zoom=10,
            pitch=0,
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                data=map_data,
                get_position='[lon, lat]',
                get_color='[200, 30, 0, 160]',
                get_radius=200,
                pickable=True,
            ),
            pdk.Layer(
                'LineLayer',
                data=pd.DataFrame({
                    'source_lat': [rest_lat],
                    'source_lon': [rest_lon],
                    'target_lat': [deliv_lat],
                    'target_lon': [deliv_lon],
                }),
                get_source_position='[source_lon, source_lat]',
                get_target_position='[target_lon, target_lat]',
                get_color='[0, 0, 255]',
                get_width=4,
            )
        ]
    ))
