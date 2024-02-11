import streamlit as st
import pandas as pd
import joblib
import os

def load_model(model_file_path):
    return joblib.load(model_file_path)

def load_data(data_file_path):
    return pd.read_csv(data_file_path)

def predict_heart_attack(model, data, physicalhealthdays, mentalhealthdays, physicalactivities, sleephours, hadstroke, hadasthma, hadcopd, haddepressivedisorder, difficultyconcentrating, difficultywalking, bmi, alcoholdrinkers, had_diabetes, age_column, received_tetanus, received_not, received_tdap, smoking_never_smoked, smoking_current_smoker, smoking_former_smoker):
    age_columns = [col for col in data.columns if col.startswith('age_Age')]
    age_encoded = [1 if col == age_column else 0 for col in age_columns]

    prediction = model.predict([[
        physicalhealthdays, mentalhealthdays, physicalactivities, sleephours, hadstroke, hadasthma,
        hadcopd, haddepressivedisorder, difficultyconcentrating, difficultywalking, bmi, alcoholdrinkers,
        had_diabetes
    ] + age_encoded + [received_tetanus, received_not, received_tdap, smoking_never_smoked, smoking_current_smoker, smoking_former_smoker]])
    
    return prediction

def main():
    # Set page configuration
    st.set_page_config(
        page_title="CardioRisk Analyzer",
        page_icon="❤️",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    # Get the absolute path to the directory containing this script
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Load the pre-trained model
    model_file_path = os.path.join(script_directory, "random_forest_model.pkl")
    random_forest = load_model(model_file_path)

    # Load the preprocessed data
    data_file_path = os.path.join(script_directory, "heart_disease_preprocessed_data.csv")
    heart_df = load_data(data_file_path)

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

    result = None  # Initialize result variable

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
        result = predict_heart_attack(random_forest, heart_df, physicalhealthdays, mentalhealthdays, physicalactivities, sleephours, hadstroke, hadasthma, hadcopd, haddepressivedisorder, difficultyconcentrating, difficultywalking, bmi, alcoholdrinkers, had_diabetes, age_column, received_tetanus, received_not, received_tdap, smoking_never_smoked, smoking_current_smoker, smoking_former_smoker)
        
    if result == 1:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("<p style='text-align:center; background-color:red; color:white; padding:10px;'>You are at risk of heart disease.</p>", unsafe_allow_html=True)
            st.image("un_healthy.jpg", width=300)
    elif result == 0:  # Check for 0 as well
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("<p style='text-align:center; background-color:#6EBB62; color:white; padding:10px;'>You are not at risk of heart disease.</p>", unsafe_allow_html=True)
            st.image("healthy_heart.png", width=300)

if __name__ == "__main__":
    main()
