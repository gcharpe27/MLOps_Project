import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_project_model_v1.joblib")
model = joblib.load(model_path)

# Design Application UI 
st.title("Tourism Package Purchase Predictor")
st.write("""
This application predicts the likelihood of a customer purchasing the Wellness Tourism Package
based on their details and interaction data.
""")

st.subheader("Customer Details")

# Input widgets for numerical features
age = st.number_input("Age", min_value=18, max_value=100, value=30)
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1.0, max_value=60.0, value=10.0)
num_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=1)
num_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3)
preferred_property_star = st.selectbox("Preferred Property Star Rating", [1, 2, 3, 4, 5], index=2) # Assuming 3 is default
num_trips = st.number_input("Number of Trips Annually", min_value=0, max_value=50, value=5)
passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
own_car = st.selectbox("Owns Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
num_children_visiting = st.number_input("Number of Children Visiting (below 5)", min_value=0, max_value=5, value=0)
monthly_income = st.number_input("Monthly Income", min_value=0.0, value=50000.0, step=1000.0)
pitch_satisfaction_score = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5], index=3) # Assuming 4 is default

# Input widgets for categorical features
type_of_contact = st.selectbox("Type of Contact", ['Self Inquiry', 'Company Invited'])
city_tier = st.selectbox("City Tier", [1, 2, 3])
occupation = st.selectbox("Occupation", ['Salaried', 'Small Business', 'Other', 'Free Lancer', 'Large Business'])
gender = st.selectbox("Gender", ['Male', 'Female'])
marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced', 'Unmarried'])
product_pitched = st.selectbox("Product Pitched", ['Wellness', 'Resort', 'Other', 'Adventure', 'Road trip', 'Mountain', 'Water sports'])
designation = st.selectbox("Designation", ['Manager', 'Executive', 'Senior Manager', 'AVP', 'VP', 'Director'])

# Collect all inputs into a dictionary
input_data = {
    'Age': age,
    'DurationOfPitch': duration_of_pitch,
    'NumberOfPersonVisiting': num_person_visiting,
    'NumberOfFollowups': num_followups,
    'PreferredPropertyStar': preferred_property_star,
    'NumberOfTrips': num_trips,
    'Passport': passport,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': num_children_visiting,
    'MonthlyIncome': monthly_income,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'TypeofContact': type_of_contact,
    'CityTier': city_tier,
    'Occupation': occupation,
    'Gender': gender,
    'MaritalStatus': marital_status,
    'ProductPitched': product_pitched,
    'Designation': designation
}

# Convert input data to a DataFrame
input_df = pd.DataFrame([input_data])

if st.button("Predict Purchase"):    
    # Make prediction
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[:, 1]

    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.success(f"The customer is likely to purchase the package (Probability: {prediction_proba[0]:.2f})")
    else:
        st.info(f"The customer is unlikely to purchase the package (Probability: {prediction_proba[0]:.2f})")
    
