import streamlit as st
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("KNN_tas.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="KNN Purchase Prediction", layout="centered")

st.title("🛒 Social Network Ads Prediction")
st.write("Predict whether a user will purchase a product based on Age and Salary.")

# Input fields
age = st.number_input("Enter Age", min_value=1, max_value=100, value=30)
salary = st.number_input("Enter Estimated Salary", min_value=1000, max_value=200000, value=50000, step=1000)

if st.button("Predict"):
    # Prepare input as array
    input_data = np.array([[age, salary]])

    # Scale input
    scaled_data = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_data)[0]

    # Show result
    if prediction == 1:
        st.success("✅ The user is likely to PURCHASE the product.")
    else:
        st.warning("❌ The user is NOT likely to purchase the product.")

    # Optional: show probability if available
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(scaled_data)[0][prediction]
        st.write(f"Confidence: **{prob*100:.2f}%**")

st.markdown("---")
st.caption("Model: KNN | Features: Age, Estimated Salary")
