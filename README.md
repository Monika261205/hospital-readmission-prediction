# 🏥 Hospital Readmission Risk Prediction

This project predicts whether a patient is likely to be readmitted within 30 days using Machine Learning.

## 🚀 Features
- Patient readmission prediction using Random Forest model
- Interactive Streamlit dashboard
- Modern UI with probability display
- Voice output prediction (Text-to-Speech)

## 🛠 Tools & Technologies
- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- gTTS (voice output)

## 📂 Project Structure
- `notebooks/` → EDA + model training
- `model/` → trained ML model (.pkl)
- `app.py` → Streamlit app

## ▶ How to Run
```bash
pip install -r requirements.txt
streamlit run app.py