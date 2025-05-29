
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('simulated_used_cars_150.csv')

# Feature engineering
df['car_age'] = 2024 - df['year']
df.drop(['year', 'model', 'cylinders'], axis=1, inplace=True)
df.dropna(inplace=True)

# One-hot encoding
df = pd.get_dummies(df, columns=['manufacturer', 'condition', 'fuel', 'transmission', 'type'], drop_first=True)

# Split data
X = df.drop('price', axis=1)
y = df['price']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Streamlit UI
st.title("🚗 Used Car Price Estimator")

st.header("Enter Car Details")

car_age = st.slider('Car Age (years)', 1, 25, 5)
odometer = st.slider('Odometer (miles)', 20000, 200000, 60000, step=5000)

manufacturer = st.selectbox('Manufacturer', sorted([col.replace("manufacturer_", "") for col in df.columns if col.startswith("manufacturer_")]))
condition = st.selectbox('Condition', sorted([col.replace("condition_", "") for col in df.columns if col.startswith("condition_")]))
fuel = st.selectbox('Fuel Type', sorted([col.replace("fuel_", "") for col in df.columns if col.startswith("fuel_")]))
transmission = st.selectbox('Transmission', sorted([col.replace("transmission_", "") for col in df.columns if col.startswith("transmission_")]))
car_type = st.selectbox('Car Type', sorted([col.replace("type_", "") for col in df.columns if col.startswith("type_")]))

# Input vector construction
input_data = {col: 0 for col in X.columns}
input_data['car_age'] = car_age
input_data['odometer'] = odometer
input_data[f'manufacturer_{manufacturer}'] = 1
input_data[f'condition_{condition}'] = 1
input_data[f'fuel_{fuel}'] = 1
input_data[f'transmission_{transmission}'] = 1
input_data[f'type_{car_type}'] = 1

# Convert to DataFrame and predict
input_df = pd.DataFrame([input_data])
predicted_price = model.predict(input_df)[0]

st.subheader(f"💰 Estimated Price: ${predicted_price:,.2f}")
