import streamlit as st
import numpy as np
import joblib

model = joblib.load('random_forest.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Heart Disease Risk Predictor")
st.write("Enter patient details to estimate heart disease risk.")

age = st.slider("Age", 20, 100, 50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
chol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
restecg = st.selectbox("Resting ECG Result", [0, 1, 2])
thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1)
slope = st.selectbox("Slope of ST Segment", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (0-3)", [0, 1, 2, 3])

sex_val = 1 if sex == "Male" else 0
fbs_val = 1 if fbs == "Yes" else 0
exang_val = 1 if exang == "Yes" else 0

if st.button("Predict"):
    features = np.array([[age, sex_val, cp, trestbps, chol, fbs_val,
                           restecg, thalach, exang_val, oldpeak,
                           slope, ca, thal]])
    features_scaled = scaler.transform(features)
    
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]
    
    if prediction == 1:
        st.error(f"⚠️ High risk of heart disease (Probability: {probability:.1%})")
    else:
        st.success(f"✅ Low risk of heart disease (Probability: {probability:.1%})")
    
    st.caption("This tool is for educational purposes only and is not a medical diagnosis.")