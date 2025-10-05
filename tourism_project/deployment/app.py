import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="sivakmr4-6-3/Tourism_Package_Model", filename="best_package_prediction_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Prediction
st.title("Tourism Package Prediction App")
st.write("""
The primary objective is to build a model that predicts whether a customer will purchase the newly introduced Wellness Tourism Package before contacting them.
""")

# User input
age = st.slider("Age", min_value=18, max_value=100, value=30)
typeof_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Small Business", "Large Business", "Unemployed"])
gender = st.selectbox("Gender", ["Male", "Female"])
number_of_person_visiting = st.number_input("Number of People Visiting", min_value=1, value=1)
preferred_property_star = st.selectbox("Preferred Hotel Star Rating", [3, 4, 5])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
number_of_trips = st.number_input("Number of Trips per Year", min_value=0, value=1)
passport = st.checkbox("Do you have a passport?")
own_car = st.checkbox("Do you own a car?")
number_of_children_visiting = st.number_input("Number of Children (below age 5) Visiting", min_value=0, value=0)
designation = st.text_input("Designation in Current Organization")
monthly_income = st.number_input("Monthly Income", min_value=1000, value=10000)
pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
product_pitched = st.text_input("Type of Product Pitched")
number_of_followups = st.number_input("Number of Follow-ups", min_value=0, value=1)
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1, value=15)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': typeof_contact,
    'CityTier': city_tier,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': number_of_person_visiting,
    'PreferredPropertyStar': preferred_property_star,
    'MaritalStatus': marital_status,
    'NumberOfTrips': number_of_trips,
    'Passport': 1 if passport else 0,
    'OwnCar': 1 if own_car else 0,
    'NumberOfChildrenVisiting': number_of_children_visiting,
    'Designation': designation,
    'MonthlyIncome': monthly_income,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'ProductPitched': product_pitched,
    'NumberOfFollowups': number_of_followups,
    'DurationOfPitch': duration_of_pitch
}])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    result = "this customer will likely purchase a package" if prediction == 1 else "this customer will likely NOT purchase a package"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
