import streamlit as st
import pandas as pd
import joblib
from gtts import gTTS
import os

# Load trained model
model = joblib.load("model/readmission_model.pkl")

# Page config
st.set_page_config(page_title="Hospital Readmission Prediction", layout="wide")

# Custom CSS for Modern UI
st.markdown("""
<style>
body {
    background-color: #f5f7fb;
}

.main {
    background-color: #f5f7fb;
}

h1 {
    color: #1f2937;
    font-family: 'Segoe UI', sans-serif;
}

div.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    font-size: 16px;
    font-weight: bold;
    border: none;
}

div.stButton > button:hover {
    background-color: #1d4ed8;
    color: white;
}

.card {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.metric-box {
    background: linear-gradient(90deg, #2563eb, #4f46e5);
    padding: 18px;
    border-radius: 15px;
    color: white;
    font-size: 18px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 15px;
}

.footer {
    text-align: center;
    padding: 10px;
    font-size: 14px;
    color: gray;
}
</style>
""", unsafe_allow_html=True)

# Title Card
st.markdown("""
<div class="card">
    <h1>🏥 Hospital Readmission Risk Prediction</h1>
    <p style="font-size:16px; color:gray;">
        Predict whether a patient is at risk of being readmitted within 30 days using Machine Learning.
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar Inputs
st.sidebar.header("📌 Patient Information")

age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=45)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
insurance = st.sidebar.selectbox("Insurance Type", ["Private", "Medicare", "Medicaid", "Uninsured"])
socio_score = st.sidebar.slider("Socioeconomic Risk Score", 0.0, 10.0, 5.0)
prev_adm_6m = st.sidebar.number_input("Previous Admissions (Last 6 Months)", min_value=0, value=1)
prev_readm_1y = st.sidebar.number_input("Previous Readmissions (Last 1 Year)", min_value=0, value=0)
time_since_discharge = st.sidebar.number_input("Time Since Last Discharge (days)", min_value=0, value=30)
length_stay = st.sidebar.number_input("Length of Stay (days)", min_value=1, value=5)
admission_type = st.sidebar.selectbox("Admission Type", ["Emergency", "Elective", "Urgent"])
diagnosis_group = st.sidebar.selectbox("Primary Diagnosis Group", ["Cardiology", "Diabetes", "Respiratory", "Neurology", "Other"])
comorbidity_index = st.sidebar.slider("Comorbidity Index", 0, 10, 3)
chronic_count = st.sidebar.number_input("Chronic Disease Count", min_value=0, value=1)
icu_flag = st.sidebar.selectbox("ICU Stay Flag", [0, 1])
severity = st.sidebar.slider("Severity Score", 0.0, 10.0, 5.0)
hba1c = st.sidebar.slider("HbA1c Level", 3.0, 15.0, 7.0)
creatinine = st.sidebar.slider("Creatinine Level", 0.1, 10.0, 1.0)
hemoglobin = st.sidebar.slider("Hemoglobin Level", 5.0, 20.0, 13.0)
bp = st.sidebar.slider("Average Systolic BP", 80, 200, 120)
med_count = st.sidebar.number_input("Number of Medications", min_value=0, value=5)
med_change = st.sidebar.number_input("Medication Change Count", min_value=0, value=1)
high_risk_med = st.sidebar.selectbox("High Risk Medication Flag", [0, 1])
followup = st.sidebar.selectbox("Followup Appointment Scheduled", [0, 1])
discharge = st.sidebar.selectbox("Discharge Disposition", ["Home", "Rehab", "Nursing Facility", "Other"])
adherence = st.sidebar.slider("Medication Adherence Score", 0.0, 10.0, 7.0)

# Input DataFrame (must match training dataset column names exactly)
input_data = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "Insurance_Type": [insurance],
    "Socioeconomic_Risk_Score": [socio_score],
    "Previous_Admissions_6M": [prev_adm_6m],
    "Previous_Readmissions_1Y": [prev_readm_1y],
    "Time_Since_Last_Discharge": [time_since_discharge],
    "Length_of_Stay": [length_stay],
    "Admission_Type": [admission_type],
    "Primary_Diagnosis_Group": [diagnosis_group],
    "Comorbidity_Index": [comorbidity_index],
    "Chronic_Disease_Count": [chronic_count],
    "ICU_Stay_Flag": [icu_flag],
    "Severity_Score": [severity],
    "HbA1c_Level": [hba1c],
    "Creatinine_Level": [creatinine],
    "Hemoglobin_Level": [hemoglobin],
    "Average_Systolic_BP": [bp],
    "Number_of_Medications": [med_count],
    "Medication_Change_Count": [med_change],
    "High_Risk_Medication_Flag": [high_risk_med],
    "Followup_Appointment_Scheduled": [followup],
    "Discharge_Disposition": [discharge],
    "Medication_Adherence_Score": [adherence]
})

# Layout with columns
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("<div class='card'><h3>📌 Patient Data Preview</h3></div>", unsafe_allow_html=True)
    st.dataframe(input_data, use_container_width=True)

with col2:
    st.markdown("<div class='card'><h3>📊 Prediction Dashboard</h3></div>", unsafe_allow_html=True)
    st.info("Fill patient details from the sidebar and click **Predict Readmission Risk**.")

# Prediction button (Sidebar)
if st.sidebar.button("🔍 Predict Readmission Risk"):
    prob = model.predict_proba(input_data)[0][1]
    threshold = 0.7

    # Voice message based on result
    if prob >= threshold:
        voice_text = "Warning. This patient has high risk of hospital readmission within thirty days."
    else:
        voice_text = "This patient has low risk of hospital readmission within thirty days."

    # Convert text to speech and save MP3
    tts = gTTS(text=voice_text, lang="en")
    tts.save("prediction.mp3")

    st.markdown(f"""
    <div class="metric-box">
        Readmission Probability: {prob*100:.2f}%
    </div>
    """, unsafe_allow_html=True)

    st.progress(int(prob * 100))

    if prob >= threshold:
        st.markdown("""
        <div class="card">
            <h2 style="color:red;">⚠ HIGH RISK</h2>
            <p style="font-size:16px;">
                This patient is likely to be readmitted within <b>30 days</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card">
            <h2 style="color:green;">✅ LOW RISK</h2>
            <p style="font-size:16px;">
                This patient is unlikely to be readmitted within <b>30 days</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Voice output section
    st.subheader("🔊 Voice Output")
    st.audio("prediction.mp3")

# Footer
st.markdown("""
<div class="footer">
    Made with ❤️ using Streamlit | Hospital Readmission Prediction Project
</div>
""", unsafe_allow_html=True)