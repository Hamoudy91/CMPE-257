import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streamlit app
st.title('Crop Yield Prediction App')

st.write("""
## Predict the Crop Yield
Input the average temperature, rainfall, and pesticide usage to predict the crop yield in hectograms per hectare.
""")

# Input fields
avg_temp = st.number_input('Average Temperature (°C)', min_value=-10.0, max_value=50.0, value=25.0)
rainfall = st.number_input('Average Rainfall (mm per year)', min_value=0.0, max_value=3000.0, value=500.0)
pesticides = st.number_input('Pesticides Usage (tonnes)', min_value=0.0, max_value=1000.0, value=100.0)

# Feature engineering for input
temp_rain_interaction = avg_temp * rainfall
pesticides_rain_interaction = pesticides * rainfall
pesticides_temp_interaction = pesticides * avg_temp
temp_squared = avg_temp ** 2

# Create a DataFrame for the input features
input_data = {
    'average_rain_fall_mm_per_year': [rainfall],
    'pesticides_tonnes': [pesticides],
    'avg_temp': [avg_temp],
    'temp_rain_interaction': [temp_rain_interaction],
    'pesticides_rain_interaction': [pesticides_rain_interaction],
    'pesticides_temp_interaction': [pesticides_temp_interaction],
    'temp_squared': [temp_squared]
}

features = pd.DataFrame(input_data)

# Scale the features
scaled_features = scaler.transform(features)

# Predict button
if st.button('Predict Crop Yield'):
    prediction = model.predict(scaled_features)
    st.write(f'Predicted Crop Yield: {prediction[0]:.2f} hg/ha')
    ''