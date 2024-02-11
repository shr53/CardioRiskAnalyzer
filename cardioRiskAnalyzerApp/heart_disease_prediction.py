import streamlit as st
import pandas as pd
import joblib
import os

def load_model(model_file_path):
    return joblib.load(model_file_path)

def load_data(data_file_path):
    return pd.read_csv(data_file_path)

def predict_heart_attack(model, data, physicalhealthdays, mentalhealthdays, physicalactivities, sleephours, hadstroke, hadasthma, hadcopd, haddepressivedisorder, difficultyconcentrating, difficultywalking, bmi, alcoholdrinkers, had_diabetes, age_range, received_tetanus, received_not, received_tdap, smoking_never_smoked, smoking_current_smoker, smoking_former_smoker):
    # Map "Yes" and "No" to numerical values
    yes_no_mapping = {"No": 0, "Yes": 1}

    # Map categorical features to numerical values
    physicalactivities = yes_no_mapping[physicalactivities]
    hadstroke = yes_no_mapping[hadstroke]
    hadasthma = yes_no_mapping[hadasthma]
    hadcopd = yes_no_mapping[hadcopd]
    haddepressivedisorder = yes_no_mapping[haddepressivedisorder]
    difficultyconcentrating = yes_no_mapping[difficultyconcentrating]
    difficultywalking = yes_no_mapping[difficultywalking]
    alcoholdrinkers = yes_no_mapping[alcoholdrinkers]
    had_diabetes = yes_no_mapping[had_diabetes]

    # Encode age range
    age_column = 'age_Age ' + age_range.replace(" ", "")
    age_columns = [col for col in data.columns if col.startswith('age_Age')]
    age_encoded = [1 if col == age_column else 0 for col in age_columns]

    # Map received_vaccine and smoking_status to numerical values
    received_vaccine_mapping = {"Tetanus": 1, "Not Received": 0, "TDAP": 1}
    received_tetanus = received_vaccine_mapping["Tetanus"] if received_tetanus == 1 else 0
    received_not = received_vaccine_mapping["Not Received"] if received_not == 1 else 0
    received_tdap = received_vaccine_mapping["TDAP"] if received_tdap == 1 else 0

    smoking_status_mapping = {"Never Smoked": 0, "Current Smoker": 1, "Former Smoker": 1}
    smoking_status = smoking_status_mapping["Never Smoked"] if smoking_never_smoked == 1 else 0
    smoking_status += smoking_status_mapping["Current Smoker"] if smoking_current_smoker == 1 else 0
    smoking_status += smoking_status_mapping["Former Smoker"] if smoking_former_smoker == 1 else 0

    # Pass the features to the prediction function
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
    
    # User input fields
    physicalhealthdays = st.sidebar.slider("Physical Health Days", min_value=0, max_value=30, value=15)
    mentalhealthdays = st.sidebar.slider("Mental Health Days", min_value=0, max_value=30, value=15)
    sleephours = st.sidebar.slider("Sleep Hours", min_value=0, max_value=30, value=15)
    physicalactivities = st.sidebar.selectbox("Physical Activities", ["No", "Yes"])
    hadstroke = st.sidebar.selectbox("Had Stroke", ["No", "Yes"])
    hadasthma = st.sidebar.selectbox("Had Asthma", ["No", "Yes"])
    hadcopd = st.sidebar.selectbox("Had COPD", ["No", "Yes"])
    haddepressivedisorder = st.sidebar.selectbox("Had Depressive Disorder", ["No", "Yes"])
    difficultyconcentrating = st.sidebar.selectbox("Difficulty Concentrating", ["No", "Yes"])
    difficultywalking = st.sidebar.selectbox("Difficulty Walking", ["No", "Yes"])
    bmi = st.sidebar.text_input("BMI (between 12 and 90)")
    alcoholdrinkers = st.sidebar.selectbox("Alcohol Drinkers", ["No", "Yes"])
    had_diabetes = st.sidebar.selectbox("Had Diabetes", ["No", "Yes"])
    age_range = st.sidebar.selectbox("Age Range", ["18 to 24", "25 to 29", "30 to 34", "35 to 39", "40 to 44", "45 to 49", "50 to 54", "55 to 59", "60 to 64", "65 to 69", "70 to 74", "75 to 79", "80 or older"])
    received_vaccine = st.sidebar.selectbox("Received Vaccine", ["Tetanus", "Not Received", "TDAP"])
    smoking_status = st.sidebar.selectbox("Smoking Status", ["Never Smoked", "Current Smoker", "Former Smoker"])

    result = None  # Initialize result variable

    if st.sidebar.button("Predict"):
        # Check if BMI input is valid
        if bmi == "Enter your BMI here":
            st.error("Please enter a valid BMI.")
        else:
            # Assign a value to received_vaccine first
            received_vaccine_mapping = {"Tetanus": 1, "Not Received": 0, "TDAP": 1}
            received_vaccine_val = received_vaccine_mapping[received_vaccine]
            
            # Calculate other variables based on received_vaccine value
            received_tetanus = 1 if received_vaccine == "Tetanus" else 0
            received_not = 1 if received_vaccine == "Not Received" else 0
            received_tdap = 1 if received_vaccine == "TDAP" else 0
            smoking_never_smoked = 1 if smoking_status == "Never Smoked" else 0
            smoking_current_smoker = 1 if smoking_status == "Current Smoker" else 0
            smoking_former_smoker = 1 if smoking_status == "Former Smoker" else 0

            result = predict_heart_attack(random_forest, heart_df, physicalhealthdays, mentalhealthdays, physicalactivities, sleephours, hadstroke, hadasthma, hadcopd, haddepressivedisorder, difficultyconcentrating, difficultywalking, float(bmi), alcoholdrinkers, had_diabetes, age_range, received_tetanus, received_not, received_tdap, smoking_never_smoked, smoking_current_smoker, smoking_former_smoker)
        
    if result is not None:
        col1, col2, col3 = st.columns([1, 2, 1])
        if result == 1:
            with col2:
                st.markdown("<p style='text-align:center; background-color:red; color:white; padding:10px;'>You are at risk of heart disease.</p>", unsafe_allow_html=True)
        elif result == 0:
            with col2:
                st.markdown("<p style='text-align:center; background-color:#6EBB62; color:white; padding:10px;'>You are not at risk of heart disease.</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()