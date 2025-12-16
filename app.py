import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Math Score Predictor", page_icon="📘", layout="centered")

@st.cache_resource
def load_artifacts():
    with open("rf_math_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)

    return model, encoders

def encode_input(encoders, data_dict):
    """
    encoders diasumsikan dict: {column_name: LabelEncoder}
    """
    encoded = {}
    for col, val in data_dict.items():
        if col in encoders:
            le = encoders[col]
            # validasi nilai ada di classes_
            if val not in set(le.classes_):
                raise ValueError(f"Nilai '{val}' tidak dikenal untuk kolom '{col}'. Pilih dari: {list(le.classes_)}")
            encoded[col] = int(le.transform([val])[0])
        else:
            # kolom numerik (reading/writing)
            encoded[col] = val
    return encoded

st.title("📘 Prediksi Math Score (Random Forest)")
st.caption("Input: profil siswa + reading score + writing score → Output: prediksi math score")

# Load model & encoders
try:
    model, encoders = load_artifacts()
except Exception as e:
    st.error("Gagal memuat model/encoder. Cek requirements.txt (versi numpy/sklearn) dan file .pkl.")
    st.exception(e)
    st.stop()

# Ambil pilihan kategori dari encoder (lebih aman daripada hardcode)
def options(col, fallback):
    if col in encoders:
        return list(encoders[col].classes_)
    return fallback

gender_opt = options("gender", ["female", "male"])
race_opt = options("race/ethnicity", ["group A", "group B", "group C", "group D", "group E"])
parent_opt = options(
    "parental level of education",
    ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
)
lunch_opt = options("lunch", ["standard", "free/reduced"])
prep_opt = options("test preparation course", ["none", "completed"])

with st.form("input_form"):
    gender = st.selectbox("gender", gender_opt)
    race = st.selectbox("race/ethnicity", race_opt)
    parent = st.selectbox("parental level of education", parent_opt)
    lunch = st.selectbox("lunch", lunch_opt)
    prep = st.selectbox("test preparation course", prep_opt)

    reading = st.number_input("reading score", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
    writing = st.number_input("writing score", min_value=0.0, max_value=100.0, value=70.0, step=1.0)

    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        raw = {
            "gender": gender,
            "race/ethnicity": race,
            "parental level of education": parent,
            "lunch": lunch,
            "test preparation course": prep,
            "reading score": float(reading),
            "writing score": float(writing),
        }

        encoded = encode_input(encoders, raw)
        X = pd.DataFrame([encoded])

        pred = model.predict(X)[0]
        pred = float(pred)

        st.success(f"✅ Predicted math score: **{pred:.2f}**")

    except Exception as e:
        st.error("Terjadi error saat prediksi. Kemungkinan mismatch nama kolom / encoder / versi library.")
        st.exception(e)
