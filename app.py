import streamlit as st
import pickle
import pandas as pd

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load feature names
with open("features.pkl", "rb") as f:
    feature_names = pickle.load(f)

st.title("Employee Attrition Prediction App 🚀")

st.write("Enter employee details below:")

# Collect user input dynamically
input_data = {}
for col in feature_names:
    # Decide categorical vs numeric
    if col.lower() in ["businesstravel", "department", "educationfield", "gender", "jobrole", "maritalstatus", "overtime"]:
        # categorical → selectbox with common options
        if col.lower() == "businesstravel":
            input_data[col] = st.selectbox(col, ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
        elif col.lower() == "department":
            input_data[col] = st.selectbox(col, ["Sales", "Research & Development", "Human Resources"])
        elif col.lower() == "educationfield":
            input_data[col] = st.selectbox(col, ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"])
        elif col.lower() == "gender":
            input_data[col] = st.selectbox(col, ["Male", "Female"])
        elif col.lower() == "jobrole":
            input_data[col] = st.selectbox(col, [
                "Sales Executive", "Research Scientist", "Laboratory Technician",
                "Manager", "Manufacturing Director", "Healthcare Representative",
                "Sales Representative", "Human Resources", "Technical Architect"
            ])
        elif col.lower() == "maritalstatus":
            input_data[col] = st.selectbox(col, ["Single", "Married", "Divorced"])
        elif col.lower() == "overtime":
            input_data[col] = st.selectbox(col, ["Yes", "No"])
    else:
        # numeric → number_input
        input_data[col] = st.number_input(col, value=0)

# Convert dict to DataFrame
input_df = pd.DataFrame([input_data])

if st.button("Predict"):
    prediction = model.predict(input_df)
    st.success(f"✅ Prediction: {prediction[0]}")
