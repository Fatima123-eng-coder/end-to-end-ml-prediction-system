"""Heart disease classification — classroom demo ONLY, not a diagnostic tool."""
from pathlib import Path
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="Heart Disease Prediction — Demo", page_icon="🫀", layout="centered")

@st.cache_resource
def load_bundle():
    path = Path(__file__).resolve().parent / "heart_disease_model.pkl"
    return joblib.load(path)

st.title("🫀 Heart Disease Prediction System")
st.warning("Educational machine-learning demonstration only. Not medical advice, diagnosis, or screening.")
st.write("This UCI Cleveland classroom model uses exactly four input features.")

CP_OPTIONS = ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"]
with st.form("heart_form"):
    age = st.number_input("Age (years)", min_value=18, max_value=100, value=54, step=1)
    sex = st.selectbox("Sex (dataset coding)", ["Female", "Male"])
    cp = st.selectbox("Chest pain type", CP_OPTIONS)
    thalach = st.number_input("Maximum heart rate achieved (bpm)", min_value=50, max_value=250, value=140, step=1)
    submitted = st.form_submit_button("Run prediction", type="primary")

if submitted:
    try:
        bundle = load_bundle()
        ordered_features = bundle["features"]
        sample = pd.DataFrame([{"age": age, "sex": sex, "cp": cp, "thalach": thalach}], columns=ordered_features)
        for field in ("sex", "cp"):
            sample[field] = bundle["encoders"][field].transform(sample[field])
        prediction = int(bundle["model"].predict(sample)[0])
        if prediction == 1:
            st.info("Model result: Class 1 — heart disease presence in the dataset's definition.")
        else:
            st.info("Model result: Class 0 — no heart disease presence in the dataset's definition.")
        st.write("**This prediction does not establish whether someone has heart disease.**")
        with st.expander("Encoded inputs (four features)"):
            st.dataframe(sample)
    except FileNotFoundError:
        st.error("Missing heart_disease_model.pkl. Train Task 2 notebook and upload the model file.")

st.caption("Data: UCI Heart Disease, Cleveland processed subset (CC BY 4.0). Features: age, sex, cp, thalach.")
