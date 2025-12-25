import streamlit as st
import pandas as pd
import joblib

# Load model & encoder (saved from notebook)
model = joblib.load("rf_model.joblib")
encoder = joblib.load("encoder.joblib")

# Load dataset for dropdown values only
df = pd.read_csv("US_medical_negligence.csv")

st.title("Medical Malpractice Claim Predictor")

# Inputs
amount = st.number_input("Claim Amount")
age = st.number_input("Patient Age", min_value=0, max_value=100)
private_attorney = st.selectbox("Private Attorney", [0,1])
marital_status = st.selectbox("Marital Status", [0,1,2,3,4])
specialty = st.selectbox("Specialty", df['Specialty'].unique())
insurance = st.selectbox("Insurance", df['Insurance'].unique())
gender = st.selectbox("Gender", df['Gender'].unique())

if st.button("Predict"):
    # Create input dataframe
    input_df = pd.DataFrame([[amount, age, private_attorney, marital_status, specialty, insurance, gender]],
                            columns=['Amount','Age','Private Attorney','Marital Status','Specialty','Insurance','Gender'])
    
    # Encode categorical features
    encoded_input = encoder.transform(input_df[['Specialty','Insurance','Gender']])
    encoded_input_df = pd.DataFrame(encoded_input, columns=encoder.get_feature_names_out(['Specialty','Insurance','Gender']))
    
    # Combine numeric + encoded
    final_input = pd.concat([input_df[['Amount','Age','Private Attorney','Marital Status']], encoded_input_df], axis=1)
    
    # Make prediction
    prediction = model.predict(final_input)
    st.write("High Severity Claim Likely?" , "Yes" if prediction[0]==1 else "No")
