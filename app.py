import streamlit as st
import numpy as np
from model import train_model

# Page config
st.set_page_config(page_title="Student Score Predictor", layout="centered")

# Title
st.title("🎓 Student Score Prediction (Lasso Regression)")

# Cache model to avoid retraining every refresh
@st.cache_resource
def load_model():
    model, scaler, mse, r2, coef = train_model()
    return model, scaler, mse, r2, coef

model, scaler, mse, r2, coef = load_model()

# -------------------------------
# Model Performance
# -------------------------------
st.subheader("📊 Model Performance")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"R² Score: {r2:.2f}")

# -------------------------------
# Feature Importance
# -------------------------------
st.subheader("📌 Feature Importance")
st.dataframe(coef)

# -------------------------------
# User Input
# -------------------------------
st.subheader("🔮 Predict Final Score")

col1, col2 = st.columns(2)

with col1:
    hours = st.slider("Hours Studied", 0.0, 12.0, 5.0)
    attendance = st.slider("Attendance (%)", 0.0, 100.0, 75.0)
    sleep = st.slider("Sleep Hours", 0.0, 12.0, 7.0)

with col2:
    previous = st.slider("Previous Score", 0.0, 100.0, 60.0)
    internet = st.slider("Internet Usage (hrs)", 0.0, 12.0, 3.0)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):
    try:
        input_data = np.array([[hours, attendance, sleep, previous, internet]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)

        st.success(f"🎯 Predicted Final Score: {prediction[0]:.2f}")

    except Exception as e:
        st.error(f"Error: {e}")
