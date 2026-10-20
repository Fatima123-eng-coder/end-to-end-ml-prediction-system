"""Titanic teaching demo — companion to original instructor project."""
from pathlib import Path
import pickle
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢", layout="centered")

@st.cache_resource
def load_resources():
    model_path = Path(__file__).resolve().parent / "svc_trained_model.pkl"
    with open(model_path, "rb") as handle:
        model = pickle.load(handle)
    categories = {
        "PClass": ["First", "Second", "Third"],
        "Gender": ["Male", "Female"],
        "Sibling": ["Zero", "One", "Two", "Three"],
        "Embarked": ["Southampton", "Cherbourg", "Queenstown"],
    }
    encoders = {key: LabelEncoder().fit(options) for key, options in categories.items()}
    return model, encoders

st.title("🚢 Titanic Passenger Survival Predictor")
st.write("Educational demo adapted from the course Titanic SVC example.")
with st.form("titanic_form"):
    pclass = st.selectbox("Passenger class", ["First", "Second", "Third"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    sibling = st.selectbox("Siblings/spouses on board", ["Zero", "One", "Two", "Three"])
    embarked = st.selectbox("Port of embarkation", ["Southampton", "Cherbourg", "Queenstown"])
    submitted = st.form_submit_button("Predict survival", type="primary")

if submitted:
    try:
        model, encoders = load_resources()
        values = {"PClass": pclass, "Gender": gender, "Sibling": sibling, "Embarked": embarked}
        sample = pd.DataFrame({k: [encoders[k].transform([v])[0]] for k, v in values.items()})
        prediction = int(model.predict(sample)[0])
        if prediction == 1:
            st.success("Prediction: Survived")
        else:
            st.error("Prediction: Did not survive")
        with st.expander("See encoded features"):
            st.dataframe(sample)
    except FileNotFoundError:
        st.error("Missing svc_trained_model.pkl. Train the Task 1 notebook and upload the model file to GitHub.")

st.caption("Small simulated educational dataset from the instructor's public teaching project. Not a historical or safety-critical forecast.")
