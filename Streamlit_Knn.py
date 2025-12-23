import streamlit as st
import pickle
import numpy as np

# Load model
with open("KNN_tas.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("KNN Prediction App")

st.write("Enter input values to get prediction")

# Example inputs (change based on your dataset)
f1 = st.number_input("Feature 1", value=0.0)
f2 = st.number_input("Feature 2", value=0.0)
f3 = st.number_input("Feature 3", value=0.0)
f4 = st.number_input("Feature 4", value=0.0)

if st.button("Predict"):
    input_data = np.array([[f1, f2, f3, f4]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    st.success(f"Prediction: {prediction[0]}")
