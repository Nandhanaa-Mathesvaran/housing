import streamlit as st
import pandas as pd
import numpy as np
import pickle

model = pickle.load(open("xgb_house_model.pkl", "rb"))

st.title("California Housing Price Prediction App")
st.write("Enter the details below to predict the median house value")

longitude = st.number_input("Longitude", -125.0, -114.0, -120.0)
latitude = st.number_input("Latitude", 32.0, 42.0, 35.0)
housing_median_age = st.number_input("Housing Median Age", 1, 60, 20)
total_rooms = st.number_input("Total Rooms", 1, 50000, 2000)
total_bedrooms = st.number_input("Total Bedrooms", 1, 10000, 400)
population = st.number_input("Population", 1, 50000, 1500)
households = st.number_input("Households", 1, 10000, 500)
median_income = st.number_input("Median Income (×10,000)", 0.0, 15.0, 3.5)
ocean_proximity = st.selectbox("Ocean Proximity", ["<1H OCEAN","INLAND","ISLAND","NEAR BAY","NEAR OCEAN"])

input_data = pd.DataFrame({
    'longitude':[longitude],
    'latitude':[latitude],
    'housing_median_age':[housing_median_age],
    'total_rooms':[total_rooms],
    'total_bedrooms':[total_bedrooms],
    'population':[population],
    'households':[households],
    'median_income':[median_income],
    'ocean_proximity':[ocean_proximity]
})

# Apply log transformation to match training
cols_to_log = ['total_rooms','total_bedrooms','population','households','median_income']
for col in cols_to_log:
    input_data[col] = np.log1p(input_data[col])

input_encoded = pd.get_dummies(input_data, drop_first=True)

# Ensure all columns match training
all_columns = ['longitude','latitude','housing_median_age','total_rooms','total_bedrooms',
               'population','households','median_income',
               'ocean_proximity_INLAND','ocean_proximity_ISLAND','ocean_proximity_NEAR BAY','ocean_proximity_NEAR OCEAN']
input_encoded = input_encoded.reindex(columns=all_columns, fill_value=0)

if st.button("Predict"):
    prediction = model.predict(input_encoded)[0]
    st.write(f"Predicted Median House Value: ${prediction:,.2f}")
