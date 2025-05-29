# 🚗 Used Car Price Predictor

A Streamlit app that predicts used car prices based on basic input features like age, odometer, condition, and fuel type — powered by a trained machine learning model (Random Forest).

---

## 🧠 Features

- Dropdown form inputs for:
  - Manufacturer
  - Condition
  - Fuel Type
  - Transmission
  - Car Type
- Sliders for:
  - Car age
  - Odometer mileage
- Instant price prediction using a trained Random Forest model

---

## 📊 Dataset

Based on a simulated dataset of 150 used car listings with fields like:
- Price
- Year (used to calculate car age)
- Odometer
- Manufacturer, Condition, Fuel, Transmission, Type

---

## ▶️ Run It Live

You can run this locally:

```bash
pip install streamlit
streamlit run car_price_app.py
