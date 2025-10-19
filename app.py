import streamlit as st
import pickle
import numpy as np
import pandas as pd

with open('xgb_house_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("California Housing Price Prediction App")
st.write("Enter details below to predict the median house value:")

longitude = st.number_input("Longitude", value=-120.0)
latitude = st.number_input("Latitude", value=35.0)
housing_median_age = st.number_input("Housing Median Age", min_value=1, max_value=60, value=20)
households = st.number_input("Number of Households", min_value=1, value=500)
median_income = st.number_input("Median Income", min_value=0.0, value=3.5)
rooms_per_household = st.number_input("Rooms per Household", min_value=1.0, value=5.0)
bedrooms_per_room = st.number_input("Bedrooms per Room", min_value=0.1, value=0.2)
population_per_household = st.number_input("Population per Household", min_value=1.0, value=3.0)
ocean_proximity = st.selectbox("Ocean Proximity", ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"])

input_data = pd.DataFrame({
    'longitude': [longitude],
    'latitude': [latitude],
    'housing_median_age': [housing_median_age],
    'households': [households],
    'median_income': [median_income],
    'rooms_per_household': [rooms_per_household],
    'bedrooms_per_room': [bedrooms_per_room],
    'population_per_household': [population_per_household],
    'ocean_proximity': [ocean_proximity]
})

input_data = pd.get_dummies(input_data, drop_first=True)

model_features = model.get_booster().feature_names
for col in model_features:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[model_features]


if st.button("Predict House Price"):
    prediction = model.predict(input_data)[0]
    st.success(f" Predicted Median House Value: ${prediction:,.2f}")
