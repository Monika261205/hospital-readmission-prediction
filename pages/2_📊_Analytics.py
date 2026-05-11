import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Analytics", layout="wide")

st.title("📊 Dataset Analytics Dashboard")

df = pd.read_csv("data/hospital.csv")

st.subheader("🔍 Dataset Preview")
st.dataframe(df.head(20), use_container_width=True)

st.subheader("📌 Dataset Summary")
st.write(df.describe())

st.subheader("🎯 Target Variable Distribution")

target_counts = df["Readmitted_Within_30_Days"].value_counts()

fig, ax = plt.subplots()
ax.bar(target_counts.index.astype(str), target_counts.values)
ax.set_xlabel("Readmitted Within 30 Days")
ax.set_ylabel("Count")
st.pyplot(fig)

st.subheader("📈 Age Distribution")

fig2, ax2 = plt.subplots()
ax2.hist(df["Age"], bins=20)
ax2.set_xlabel("Age")
ax2.set_ylabel("Frequency")
st.pyplot(fig2)