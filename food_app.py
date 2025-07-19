import streamlit as st
import numpy as np
from keras.models import load_model
from math import radians, sin, cos, sqrt, atan2


# ------------------------------
# background image
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
    R = 6371  # Radius of the Earth in kilometers
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("Food Delivery Time Prediction")

# Age
age = st.number_input("Delivery Person Age:", min_value= 18, max_value= 65, value= 30)

# Rating
rating = st.slider("Delivery Person Rating", min_value= 1.0, max_value= 5.0, value= 4.0, step= 0.1)

# Resturant Location
st.subheader("Restaurant Location")
rest_lat = st.number_input("Restaurant Latitude:", format="%.6f", value= 12.9715987)
rest_lon = st.number_input("Restaurant Longitude:", format="%.6f", value= 77.594566)

# Delivery Location
st.subheader("Delivery Location")
deliv_lat = st.number_input("Delivery Latitude:", format="%.6f", value= 12.2958104)
deliv_lon = st.number_input("Delivery Longitude:", format="%.6f", value= 76.6393805)

# ------------------------------
# Prediction
# ------------------------------
if st.button("Predict Delivery Time"):
    with st.spinner("Predicting delivery time..."):

        distance = calc_distance(rest_lat, rest_lon, deliv_lat, deliv_lon)

        # Prepare input data for the model
        input_data = np.array([[age, rating, distance]])
        input_reshape = input_data.reshape((1, 3, 1))

        # Make prediction
        prediction = model.predict(input_reshape, verbose=0)[0][0]

    st.success(f"Estimated Delivery Time: **{prediction:.2f} minutes**")
    st.info(f"Distance: **{distance:.2f} km**")
    st.balloons()

# ------------------------------
# Add map visualization
# ------------------------------


import pandas as pd
import pydeck as pdk

# prepare data for map
map_data = pd.DataFrame({
    'lat': [rest_lat, deliv_lat],
    'lon': [rest_lon, deliv_lon],
    'label': ['Restaurant', 'Delivery Location']
})

# Show map
st.subheader("Delivery Route Map")
st.pydeck_chart(pdk.Deck(
    initial_view_state=pdk.ViewState(
        latitude=(rest_lat + deliv_lat) / 2,
        longitude=(rest_lon + deliv_lon) / 2,
        zoom=10,
        pitch=0,
    ),
    layers=[
        # Show pickup and drop-off points
        pdk.Layer(
            'ScatterplotLayer',
            data=map_data,
            get_position='[lon, lat]',
            get_color='[200, 30, 0, 160]',
            get_radius=200,
            pickable=True,
        ),
        # Line connecting the two points
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