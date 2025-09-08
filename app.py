import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("rf_house_prices.pkl")

st.title("House Price Predictor")
st.write("Enter house details to estimate the sale price")

# Example input fields
lot_area = st.number_input("Lot Area (sq ft)", 500, 200000, 10000)
year_built = st.number_input("Year Built", 1800, 2023, 2000)
overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", 300, 6000, 1500)

# Create dataframe with the same columns the model expects
input_data = pd.DataFrame([[lot_area, year_built, overall_qual, gr_liv_area]],
                          columns=["LotArea", "YearBuilt", "OverallQual", "GrLivArea"])

# Prediction
prediction = model.predict(input_data)[0]
st.subheader(f"💰 Estimated Sale Price: ${prediction:,.2f}")
