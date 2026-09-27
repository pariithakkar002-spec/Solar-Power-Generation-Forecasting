import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load the trained model and scaler
model = joblib.load("solar_Power_generation_Forecasting_model.pkl")
scaler = joblib.load('scaler.pkl')

st.title(" Solar DC Power Forecasting App")

st.markdown("Enter the environmental conditions and date/time to predict solar power generation.")

# User Inputs
date_input = st.text_input("Enter Date and Time (YYYY-MM-DD HH:MM)", "2020-06-15 14:00")

IRRADIATION = st.number_input("IRRADIATION", min_value=0.0, step=0.1, format="%.2f")
MODULE_TEMPERATURE = st.number_input("MODULE TEMPERATURE (°C)", min_value=0.0, step=0.1, format="%.2f")
AMBIENT_TEMPERATURE = st.number_input("AMBIENT TEMPERATURE (°C)", min_value=0.0, step=0.1, format="%.2f")

if st.button("Predict DC Power"):
    try:
        input_time = datetime.strptime(date_input, "%Y-%m-%d %H:%M")
        hour = input_time.hour
        day = input_time.day
        month = input_time.month
        day_of_week = input_time.weekday()

        # Create feature input
        input_df = pd.DataFrame([[
            IRRADIATION,
            MODULE_TEMPERATURE,
            AMBIENT_TEMPERATURE,
            hour,
            day,
            month,
            day_of_week
        ]], columns=[
            'IRRADIATION',
            'MODULE_TEMPERATURE',
            'AMBIENT_TEMPERATURE',
            'HOUR',
            'DAY',
            'MONTH',
            'DAY_OF_WEEK'
        ])

        # Scale and predict
        input_scaled = scaler.transform(input_df)
        predicted_dc_power = model.predict(input_scaled)

        st.success(f"⚡ Predicted DC Power: {predicted_dc_power[0]:.4f} kW")

    except ValueError:
        st.error("Please enter date and time in format YYYY-MM-DD HH:MM")
