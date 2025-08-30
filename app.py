# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------
# 1. Load models safely
# ---------------------------

# Logistic Regression and Random Forest using joblib
import joblib

# Make sure you have all classes imported before loading
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# File paths
log_model_path = "logistic_model.pkl"
rf_model_path = "rf_model.pkl"
xgb_model_path = "xgb_model.json"  # Use XGBoost native format

# Load models
log_model = joblib.load(log_model_path)
rf_model = joblib.load(rf_model_path)

xgb_model = XGBClassifier()
xgb_model.load_model(xgb_model_path)

# ---------------------------
# 2. Streamlit App UI
# ---------------------------
st.title("Employee Attrition Prediction")

st.markdown("Enter employee details to predict attrition:")

# Example input fields (modify according to your dataset features)
age = st.number_input("Age", min_value=18, max_value=70, value=30)
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=50000, value=5000)
years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
# Add more features as needed

# Create input dataframe
input_data = pd.DataFrame({
    "Age": [age],
    "MonthlyIncome": [monthly_income],
    "YearsAtCompany": [years_at_company],
    "JobSatisfaction": [job_satisfaction]
    # Add all other features here
})

# ---------------------------
# 3. Feature Scaling (if needed)
# ---------------------------
# Assuming models were trained on scaled data
scaler = StandardScaler()
input_scaled = scaler.fit_transform(input_data)  # Or load pre-fitted scaler using joblib

# ---------------------------
# 4. Model Selection
# ---------------------------
model_choice = st.selectbox("Choose Model", ["Logistic Regression", "Random Forest", "XGBoost"])

if st.button("Predict"):
    if model_choice == "Logistic Regression":
        pred = log_model.predict(input_scaled)[0]
        proba = log_model.predict_proba(input_scaled)[0, 1]
    elif model_choice == "Random Forest":
        pred = rf_model.predict(input_scaled)[0]
        proba = rf_model.predict_proba(input_scaled)[0, 1]
    else:
        pred = xgb_model.predict(input_scaled)[0]
        proba = xgb_model.predict_proba(input_scaled)[0, 1]

    # Display prediction
    st.subheader("Prediction Result")
    st.write("Attrition: ", "Yes" if pred == 1 else "No")
    st.write("Probability of leaving: ", round(proba*100, 2), "%")

    # Optional: confusion matrix or more metrics
    # Example: placeholder plot
    cm = np.array([[50, 5],[7, 38]])  # Dummy confusion matrix
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
