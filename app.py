import numpy as np
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

model = joblib.load('heart_disease_model (1).pkl')
scaler = joblib.load('scaler (1).pkl')


def predict_heart_disease(inputs):
    inputs_scaled = scaler.transform([inputs])
    prediction = model.predict(inputs_scaled)
    return prediction[0]


def main():
    st.title('Heart Disease Prediction App')

    # Input fields for the user
    age = st.number_input('Age', min_value=0)
    sex = st.selectbox('Sex', options=[0, 1])  # Example for binary gender
    chest_pain = st.selectbox('Chest Pain Type', options=[0, 1, 2, 3])
    blood_pressure = st.number_input('Resting Blood Pressure', min_value=0)
    cholesterol = st.number_input('Serum Cholesterol', min_value=0)
    sugar = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1])
    ecg = st.selectbox('Electrocardiographic Result', options=[0, 1, 2])
    max_heart_rate = st.number_input('Maximum Heart Rate Achieved', min_value=0)
    exercise_angina = st.selectbox('Exercise Induced Angina', options=[0, 1])
    oldpeak = st.number_input('Depression Induced by Exercise Relative to Rest', min_value=0.0)
    slope = st.selectbox('Slope of the Peak Exercise ST Segment', options=[0, 1, 2])
    ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy', options=[0, 1, 2, 3])
    thal = st.selectbox('Thalassemia', options=[0, 1, 2, 3])

    # Collect inputs into a list
    inputs = [age, sex, chest_pain, blood_pressure, cholesterol, sugar, ecg, max_heart_rate,
              exercise_angina, oldpeak, slope, ca, thal]

    # Button to make prediction
    if st.button('Predict'):
        prediction = predict_heart_disease(inputs)
        if prediction == 1:
            st.success('Heart Disease Predicted')
        else:
            st.success('No Heart Disease Predicted')


if __name__ == '__main__':
    main()


