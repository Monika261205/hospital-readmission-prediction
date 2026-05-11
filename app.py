import streamlit as st
import pandas as pd
import joblib
from gtts import gTTS
import os
import matplotlib.pyplot as plt
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import io
DEMO_MODE = True
if DEMO_MODE == False:

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    def login_page():
        st.markdown("""
        <div class="card">
            <h1>🔐 Login Required</h1>
            <p style="color:gray;">Enter credentials to access the dashboard.</p>
        </div>
        """, unsafe_allow_html=True)

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        correct_username = "admin"
        correct_password = "admin123"

        if st.button("Login"):
            if username == correct_username and password == correct_password:
                st.session_state.logged_in = True
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password!")

    if not st.session_state.logged_in:
        login_page()
        st.stop()
# ------------------- LOAD MODEL -------------------
model = joblib.load("model/readmission_model.pkl")

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="Hospital Readmission Prediction", layout="wide")

# ------------------- CUSTOM CSS -------------------
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



# ------------------- TITLE -------------------
st.markdown("""
<div class="card">
    <h1>🏥 Hospital Readmission Risk Prediction</h1>
    <p style="font-size:16px; color:gray;">
        Predict whether a patient is at risk of being readmitted within 30 days using Machine Learning.
    </p>
</div>
""", unsafe_allow_html=True)

# ------------------- SIDEBAR -------------------
st.sidebar.markdown("## ℹ️ About This App")
st.sidebar.write("""
This dashboard predicts whether a patient is at risk of **hospital readmission within 30 days**.

**Model Used:** Random Forest Classifier  
**Output:** Probability + Risk Category + Voice Result + Report Download
""")

st.sidebar.markdown("---")

threshold = st.sidebar.slider("🎯 Prediction Threshold", 0.1, 0.9, 0.7)

st.sidebar.markdown("---")
st.sidebar.header("📌 Patient Information")

# Sidebar Inputs
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

numeric_cols = [
    "Age", "Socioeconomic_Risk_Score", "Previous_Admissions_6M",
    "Previous_Readmissions_1Y", "Time_Since_Last_Discharge",
    "Length_of_Stay", "Comorbidity_Index", "Chronic_Disease_Count",
    "Severity_Score", "HbA1c_Level", "Creatinine_Level",
    "Hemoglobin_Level", "Average_Systolic_BP", "Number_of_Medications",
    "Medication_Change_Count", "Medication_Adherence_Score"
]

# Input DataFrame
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

# Layout
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("<div class='card'><h3>📌 Patient Data Preview</h3></div>", unsafe_allow_html=True)
    st.dataframe(input_data, use_container_width=True)

with col2:
    st.markdown("<div class='card'><h3>📊 Prediction Dashboard</h3></div>", unsafe_allow_html=True)
    st.info("Fill patient details from the sidebar and click **Predict Readmission Risk**.")

# ------------------- GAUGE CHART FUNCTION -------------------
def draw_gauge(probability):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_aspect("equal")

    theta = np.linspace(0, np.pi, 100)
    r = 1

    ax.plot(r * np.cos(theta), r * np.sin(theta), linewidth=8)

    angle = probability * np.pi
    ax.plot([0, np.cos(angle)], [0, np.sin(angle)], linewidth=6)

    ax.text(0, -0.2, f"{probability*100:.2f}%", ha="center", fontsize=18, fontweight="bold")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    return fig

# ------------------- PDF GENERATION FUNCTION -------------------
def generate_pdf(report_df, risk_level, prob, threshold):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Hospital Readmission Prediction Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 90, f"Risk Level: {risk_level}")
    c.drawString(50, height - 110, f"Probability: {prob*100:.2f}%")
    c.drawString(50, height - 130, f"Threshold Used: {threshold:.2f}")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 170, "Patient Input Details:")

    y = height - 200
    c.setFont("Helvetica", 10)

    for col in report_df.columns:
        c.drawString(50, y, f"{col}: {report_df[col].values[0]}")
        y -= 15
        if y < 50:
            c.showPage()
            y = height - 50

    c.save()
    buffer.seek(0)
    return buffer

# ------------------- PREDICTION -------------------
if st.sidebar.button("🔍 Predict Readmission Risk"):
    prob = model.predict_proba(input_data)[0][1]
    result_label = "HIGH RISK" if prob >= threshold else "LOW RISK"

    # Voice output
    voice_text = (
        "Warning. This patient has high risk of hospital readmission within thirty days."
        if prob >= threshold
        else "This patient has low risk of hospital readmission within thirty days."
    )

    tts = gTTS(text=voice_text, lang="en")
    tts.save("prediction.mp3")

    st.markdown(f"""
    <div class="metric-box">
        Readmission Probability: {prob*100:.2f}%
    </div>
    """, unsafe_allow_html=True)

    # KPI Metrics
    st.subheader("📌 Key Metrics")
    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric("Readmission Probability", f"{prob*100:.2f}%")

    with m2:
        st.metric("Risk Level", result_label)

    with m3:
        st.metric("Threshold Used", f"{threshold:.2f}")

    # Gauge Meter
    st.subheader("🎛 Probability Gauge Meter")
    gauge_fig = draw_gauge(prob)
    st.pyplot(gauge_fig)

    # Pie Chart
    st.subheader("📊 Risk Distribution (Pie Chart)")
    pie_labels = ["Not Readmitted", "Readmitted"]
    pie_values = [1 - prob, prob]

    fig2, ax2 = plt.subplots()
    ax2.pie(pie_values, labels=pie_labels, autopct="%1.1f%%", startangle=90)
    ax2.axis("equal")
    st.pyplot(fig2)

    # Probability bar chart
    st.subheader("📈 Probability Visualization")
    fig, ax = plt.subplots()
    ax.bar(["Readmission Risk"], [prob])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    st.pyplot(fig)

    # Result card
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

    # ------------------- FEATURE IMPORTANCE -------------------
st.subheader("⭐ Top Feature Importance (Model Explanation)")

try:
    # If model is pipeline
    if hasattr(model, "named_steps"):
        rf_model = model.named_steps.get("model", model)
    else:
        rf_model = model

    feature_importance = rf_model.feature_importances_

    # Since your model is trained on raw columns (not onehot pipeline),
    # we will use the original dataset column names.
    feature_names = list(input_data.columns)

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": feature_importance
    }).sort_values(by="Importance", ascending=False).head(10)

    st.dataframe(importance_df, use_container_width=True)

    # Plot importance chart
    st.subheader("📌 Feature Importance Chart")

    fig_imp, ax_imp = plt.subplots()
    ax_imp.barh(importance_df["Feature"][::-1], importance_df["Importance"][::-1])
    ax_imp.set_xlabel("Importance Score")
    st.pyplot(fig_imp)

except Exception as e:
    st.warning("Feature importance not available. Model may not support it.")

    # Voice output section
    st.subheader("🔊 Voice Output")
    st.audio("prediction.mp3")

    # Report DataFrame
    report_df = input_data.copy()
    report_df["Readmission_Probability"] = prob
    report_df["Risk_Level"] = result_label
    report_df["Threshold_Used"] = threshold

    # Download CSV
    st.subheader("📥 Download Prediction Report")
    st.download_button(
        label="⬇ Download Report as CSV",
        data=report_df.to_csv(index=False),
        file_name="readmission_prediction_report.csv",
        mime="text/csv"
    )

    # Download PDF
    pdf_buffer = generate_pdf(report_df, result_label, prob, threshold)

    st.download_button(
        label="⬇ Download Report as PDF",
        data=pdf_buffer,
        file_name="readmission_prediction_report.pdf",
        mime="application/pdf"
    )

# Footer
st.markdown("""
<div class="footer">
    Made with ❤️ using Streamlit | Hospital Readmission Prediction Project
</div>
""", unsafe_allow_html=True)