import streamlit as st
import joblib
import pandas as pd
from utils import get_weather

model = joblib.load("irrigation_model.pkl")

st.title("🌿 WaterWise AI – Smart Irrigation System")

# ------------------ Inputs ------------------

crop = st.selectbox("Crop Type",
                    ["Wheat", "Rice", "Maize", "Cotton", "Sugarcane"])

soil = st.selectbox("Soil Type",
                    ["Sandy", "Loamy", "Clay"])

city = st.text_input("Enter City for Live Weather")

if st.button("Fetch Weather"):
    temp, humidity, rainfall = get_weather(city)
    st.success(f"Temp: {temp}°C | Humidity: {humidity}% | Rainfall: {rainfall}mm")

temperature = st.slider("Temperature (°C)", 10, 50, 30)
humidity = st.slider("Humidity (%)", 20, 100, 60)
rainfall = st.slider("Rainfall (mm)", 0, 100, 20)
soil_moisture = st.slider("Soil Moisture (%)", 5, 80, 30)

# ------------------ Prediction ------------------

if st.button("Predict Irrigation"):

    input_data = pd.DataFrame([{
        "Crop": crop,
        "Soil": soil,
        "Temperature": temperature,
        "Humidity": humidity,
        "Rainfall": rainfall,
        "Soil_Moisture": soil_moisture
    }])

    prediction = round(model.predict(input_data)[0], 2)

    # Cost estimation
    cost_per_liter = 0.05   # ₹
    estimated_cost = round(prediction * cost_per_liter, 2)

    st.subheader("🌊 Irrigation Recommendation")

    if prediction <= 5:
        st.success("No Irrigation Required")
    else:
        st.warning("Irrigation Required")

    st.write(f"Water Required: {prediction} L/m²")
    st.write(f"Estimated Irrigation Cost: ₹{estimated_cost}")

    st.image("feature_importance.png", caption="Model Feature Importance")