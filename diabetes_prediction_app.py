import streamlit as st
import numpy as np
import pickle

# Load the pre-trained model
with open('diabetes_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to make predictions
def predict_diabetes(features):
    prediction = model.predict([features])
    return 'Yes' if prediction[0] == 1 else 'No'

# Streamlit app
st.set_page_config(page_title="Diabetes Prediction App")
st.title("Diabetes Prediction App")

st.sidebar.header("Patient Data")
st.sidebar.subheader("Please enter the following details:")

# Input fields for user to enter the features
pregnancies = st.sidebar.number_input("Pregnancies (count)", min_value=0, max_value=20, value=0)
glucose = st.sidebar.number_input("Glucose (mg/dL)", min_value=0, max_value=200, value=120)
blood_pressure = st.sidebar.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=140, value=80)
skin_thickness = st.sidebar.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
insulin = st.sidebar.number_input("Insulin (mu U/mL)", min_value=0, max_value=800, value=30)
bmi = st.sidebar.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, value=25.0)
diabetes_pedigree_function = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
age = st.sidebar.number_input("Age (years)", min_value=0, max_value=120, value=30)

# Collecting input features into a list
features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]

# Button to make prediction
if st.sidebar.button("Predict"):
    outcome = predict_diabetes(features)
    st.subheader("Prediction Result")
    
    if outcome == 'Yes':
        st.write("**The model predicts that the patient is likely to have diabetes.**")
        st.write("""
        ### What does this mean?
        Based on the input data provided, the model suggests that there is a high probability that the patient has diabetes.

        ### Next Steps:
        - **Consult a healthcare professional**: It's important to consult with a doctor for a comprehensive evaluation and diagnosis.
        - **Lifestyle Changes**: Consider adopting a healthier lifestyle, including a balanced diet, regular physical activity, and maintaining a healthy weight.
        - **Monitoring**: Regular monitoring of blood glucose levels as advised by your healthcare provider.

        ### About Diabetes:
        Diabetes is a chronic condition characterized by high levels of sugar (glucose) in the blood. Early diagnosis and management are crucial to prevent complications. There are two main types of diabetes:
        - **Type 1 Diabetes**: An autoimmune condition where the body attacks insulin-producing cells.
        - **Type 2 Diabetes**: A condition where the body becomes resistant to insulin or doesn't produce enough insulin.
        """)
    else:
        st.write("**The model predicts that the patient is not likely to have diabetes.**")
        st.write("""
        ### What does this mean?
        Based on the input data provided, the model suggests that there is a low probability that the patient has diabetes.

        ### Recommendations:
        - **Healthy Lifestyle**: Continue to follow a healthy lifestyle to reduce the risk of developing diabetes in the future.
        - **Regular Check-ups**: Regular health check-ups with your healthcare provider can help you stay on top of your health.

        ### About Diabetes:
        Diabetes is a chronic condition characterized by high levels of sugar (glucose) in the blood. While this prediction is a positive sign, it is important to maintain regular health check-ups and a healthy lifestyle to ensure long-term well-being.
        """)

# To run the app, use the command: streamlit run diabetes_prediction_app.py
