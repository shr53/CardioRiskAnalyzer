import streamlit as st
import pandas as pd
import joblib
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="CardioRisk Analyzer",
    page_icon="💓",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load the pre-trained model
model_file_path = "D:/MPS Analytics/CardioRiskAnalyzer/models/random_forest_model.pkl"
random_forest = joblib.load(model_file_path)

heart_df = pd.read_csv("D:/MPS Analytics/CardioRiskAnalyzer/data/heart_disease_preprocessed_data.csv")

def predict_heart_attack(physicalhealthdays, mentalhealthdays, physicalactivities, sleephours, hadstroke, hadasthma, hadcopd, haddepressivedisorder, difficultyconcentrating, difficultywalking, bmi, alcoholdrinkers, had_diabetes, age_column, received_tetanus, received_not, received_tdap, smoking_never_smoked, smoking_current_smoker, smoking_former_smoker):
    # Load the trained model
    model_file_path = "D:/MPS Analytics/CardioRiskAnalyzer/models/random_forest_model.pkl"
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

    # User input fields
    physicalhealthdays = st.sidebar.number_input("Physical Health Days", min_value=0, max_value=365)
    mentalhealthdays = st.sidebar.number_input("Mental Health Days", min_value=0, max_value=365)
    physicalactivities = st.sidebar.number_input("Physical Activities", min_value=0, max_value=24)
    sleephours = st.sidebar.number_input("Sleep Hours", min_value=0.0, max_value=24.0, step=0.5)
    hadstroke = st.sidebar.selectbox("Had Stroke", [0, 1])
    hadasthma = st.sidebar.selectbox("Had Asthma", [0, 1])
    hadcopd = st.sidebar.selectbox("Had COPD", [0, 1])
    haddepressivedisorder = st.sidebar.selectbox("Had Depressive Disorder", [0, 1])
    difficultyconcentrating = st.sidebar.selectbox("Difficulty Concentrating", [0, 1])
    difficultywalking = st.sidebar.selectbox("Difficulty Walking", [0, 1])
    bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=50.0)
    alcoholdrinkers = st.sidebar.selectbox("Alcohol Drinkers", [0, 1])
    had_diabetes = st.sidebar.selectbox("Had Diabetes", [0, 1])
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
        
    # Show the result
        # Display prediction result
        if result == 1:  # If prediction is Yes
            st.success("Prediction: You are at risk of heart disease!")
            st.image("risk_image.jpg", caption="Heart Disease Risk", use_column_width=True)
        else:  # If prediction is No
            st.success("Prediction: You are not at risk of heart disease.")
            st.image("D:/MPS Analytics/CardioRiskAnalyzer/images/safe_image.png", caption="Healthy Heart", use_column_width=True)

if __name__ == "__main__":
    main()