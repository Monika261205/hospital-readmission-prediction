import streamlit as st

st.set_page_config(page_title="Hospital Readmission Prediction", layout="wide")

st.markdown("""
<style>
.card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card">
    <h1>🏥 Hospital Readmission Risk Prediction System</h1>
    <p style="font-size:18px; color:gray;">
        A Machine Learning based web application to predict whether a patient is likely
        to be readmitted within 30 days.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
### 📌 Pages Available:
✅ **Prediction Page** – Enter patient details and predict readmission risk  
✅ **Analytics Page** – View dataset analysis and charts  
✅ **About Page** – Project details, tech stack, and GitHub link  

---

### 🚀 How to Use
1. Go to **Prediction Page**
2. Fill patient details in sidebar
3. Click **Predict Readmission Risk**
4. Download report in PDF/CSV

---

### 👩‍💻 Tech Stack Used
- Python  
- Streamlit  
- Scikit-learn  
- RandomForest  
- Matplotlib  
- ReportLab (PDF)
- gTTS (Voice Output)
""")