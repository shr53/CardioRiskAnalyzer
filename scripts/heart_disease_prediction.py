import streamlit as st
import pandas as pd
import joblib
import streamlit as st

import os

# Print the current working directory
print("Current working directory:", os.getcwd())

# Set page configuration
st.set_page_config(
    page_title="CardioRisk Analyzer",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load the pre-trained model
model_file_path = "../models/random_forest_model.pkl"
random_forest = joblib.load(model_file_path)

heart_df = pd.read_csv("../data/heart_disease_preprocessed_data.csv")

def predict_heart_attack(physicalhealthdays, mentalhealthdays, physicalactivities, sleephours, hadstroke, hadasthma, hadcopd, haddepressivedisorder, difficultyconcentrating, difficultywalking, bmi, alcoholdrinkers, had_diabetes, age_column, received_tetanus, received_not, received_tdap, smoking_never_smoked, smoking_current_smoker, smoking_former_smoker):
    # Load the trained model
    model_file_path = "../models/random_forest_model.pkl"
    model = joblib.load(model_file_path)
    
    # Prepare the input features
    data = {
        'physicalhealthdays': physicalhealthdays,
        'mentalhealthdays': mentalhealthdays,
        'physicalactivities': physicalactivities,
        'sleephours': sleephours,
        'hadstroke': hadstroke,
        'hadasthma': hadasthma,
        'hadcopd': hadcopd,
        'haddepressivedisorder': haddepressivedisorder,
        'difficultyconcentrating': difficultyconcentrating,
        'difficultywalking': difficultywalking,
        'bmi': bmi,
        'alcoholdrinkers': alcoholdrinkers,
        'had_diabetes': had_diabetes,
    }
    
    # Encode age column
    age_columns = [col for col in heart_df.columns if col.startswith('age_Age')]
    age_encoded = [1 if col == age_column else 0 for col in age_columns]

    # Make prediction
    prediction = model.predict([list(data.values()) + age_encoded + [received_tetanus, received_not, received_tdap, smoking_never_smoked, smoking_current_smoker, smoking_former_smoker]])
    return prediction

def main():
    st.write(
        "<div style='text-align: center;'><h1><span style='font-size: 36px;'>&#128147;</span> CardioRisk Analyzer</h1></div>",
        unsafe_allow_html=True
    )

    st.write(
        "<div style='text-align: center;'><h3>This is a web application for predicting heart disease risk based on various parameters.</h3></div>",
        unsafe_allow_html=True
    )
    st.write(
        "<div style='text-align: center;'><p>Enter your parameters on the left panel, then click 'Predict' to see the result.</p></div>",
        unsafe_allow_html=True
    )
    st.sidebar.title("Parameter Selection")
    # Define a dictionary to map "Yes" and "No" to numeric values
    yes_no_mapping = {"No": 0, "Yes": 1}
    # User input fields
    physicalhealthdays = st.sidebar.number_input("Physical Health Days", min_value=0, max_value=365)
    mentalhealthdays = st.sidebar.number_input("Mental Health Days", min_value=0, max_value=365)
    physicalactivities = st.sidebar.number_input("Physical Activities", min_value=0, max_value=24)
    sleephours = st.sidebar.number_input("Sleep Hours", min_value=0.0, max_value=24.0, step=0.5)
    hadstroke = st.sidebar.selectbox("Had Stroke", ["No", "Yes"])
    hadstroke = yes_no_mapping[hadstroke]  # Map "Yes" and "No" to 1 and 0
    hadasthma = st.sidebar.selectbox("Had Asthma", ["No", "Yes"])
    hadasthma = yes_no_mapping[hadasthma]  # Map "Yes" and "No" to 1 and 0
    hadcopd = st.sidebar.selectbox("Had COPD", ["No", "Yes"])
    hadcopd = yes_no_mapping[hadcopd]  # Map "Yes" and "No" to 1 and 0
    haddepressivedisorder = st.sidebar.selectbox("Had Depressive Disorder", ["No", "Yes"])
    haddepressivedisorder = yes_no_mapping[haddepressivedisorder]  # Map "Yes" and "No" to 1 and 0
    difficultyconcentrating = st.sidebar.selectbox("Difficulty Concentrating", ["No", "Yes"])
    difficultyconcentrating = yes_no_mapping[difficultyconcentrating]  # Map "Yes" and "No" to 1 and 0
    difficultywalking = st.sidebar.selectbox("Difficulty Walking", ["No", "Yes"])
    difficultywalking = yes_no_mapping[difficultywalking]  # Map "Yes" and "No" to 1 and 0
    bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=50.0)
    alcoholdrinkers = st.sidebar.selectbox("Alcohol Drinkers", ["No", "Yes"])
    alcoholdrinkers = yes_no_mapping[alcoholdrinkers]  # Map "Yes" and "No" to 1 and 0
    had_diabetes = st.sidebar.selectbox("Had Diabetes", ["No", "Yes"])
    had_diabetes = yes_no_mapping[had_diabetes]  # Map "Yes" and "No" to 1 and 0
    age_range = st.sidebar.selectbox("Age Range", ["18 to 24", "25 to 29", "30 to 34", "35 to 39", "40 to 44", "45 to 49", "50 to 54", "55 to 59", "60 to 64", "65 to 69", "70 to 74", "75 to 79", "80 or older"])
    received_vaccine = st.sidebar.selectbox("Received Vaccine", ["Tetanus", "Not Received", "TDAP"])
    smoking_status = st.sidebar.selectbox("Smoking Status", ["Never Smoked", "Current Smoker", "Former Smoker"])

    if st.sidebar.button("Predict"):
        # Prepare the features as required for prediction
        age_column = 'age_Age ' + age_range.replace(" ", "")  # Fixing the age column format
        received_tetanus = 1 if received_vaccine == "Tetanus" else 0
        received_not = 1 if received_vaccine == "Not Received" else 0
        received_tdap = 1 if received_vaccine == "TDAP" else 0
        smoking_never_smoked = 1 if smoking_status == "Never Smoked" else 0
        smoking_current_smoker = 1 if smoking_status == "Current Smoker" else 0
        smoking_former_smoker = 1 if smoking_status == "Former Smoker" else 0

        # Pass the features to the prediction function
        result = predict_heart_attack(physicalhealthdays, mentalhealthdays, physicalactivities, sleephours, hadstroke, hadasthma, hadcopd, haddepressivedisorder, difficultyconcentrating, difficultywalking, bmi, alcoholdrinkers, had_diabetes, age_column, received_tetanus, received_not, received_tdap, smoking_never_smoked, smoking_current_smoker, smoking_former_smoker)
        # Display prediction result
        if result == 1:  # If prediction is Yes
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("<p style='text-align:center; background-color:red; color:white; padding:10px;'>You are at risk of heart disease.</p>", unsafe_allow_html=True)
                image_path1 = r"..\images\un_healthy.jpg"
                st.image(image_path1, width=300)

        else:  # If prediction is No
            #Get the image path
            image_path = "..\images\healthy_heart.png" 
            # Center-align the image using Streamlit's layout options
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("<p style='text-align:center; background-color:#6EBB62; color:white; padding:10px;'>You are not at risk of heart disease.</p>", unsafe_allow_html=True)
                # Display the image with custom width
                st.image(image_path, width=300)

if __name__ == "__main__":
    main()