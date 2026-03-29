import streamlit as st
from model import train_model

st.title("🎓 Student Score Prediction (Lasso Regression)")

# Train model
model, scaler, mse, r2, coef = train_model()

st.subheader("📊 Model Performance")
st.write(f"MSE: {mse}")
st.write(f"R² Score: {r2}")

st.subheader("📌 Feature Importance")
st.dataframe(coef)

st.subheader("🔮 Predict Final Score")

# Inputs
hours = st.slider("Hours Studied", 0, 12)
attendance = st.slider("Attendance", 0, 100)
sleep = st.slider("Sleep Hours", 0, 12)
previous = st.slider("Previous Score", 0, 100)
internet = st.slider("Internet Usage", 0, 12)

if st.button("Predict"):
    data = scaler.transform([[hours, attendance, sleep, previous, internet]])
    prediction = model.predict(data)

    st.success(f"Predicted Score: {prediction[0]:.2f}")
