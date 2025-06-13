import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model dan preprocessing
model = joblib.load("models/xgb_best.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

st.title("🚀 Aplikasi Klasifikasi Obesitas")
st.markdown("Masukkan data berikut untuk memprediksi klasifikasi obesitas:")

# Buat form input fitur
with st.form("input_form"):
    Gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    Age = st.number_input("Usia", min_value=1.0, max_value=100.0, value=25.0)
    Height = st.number_input("Tinggi Badan (meter)", min_value=1.0, max_value=2.5, value=1.70)
    Weight = st.number_input("Berat Badan (kg)", min_value=30.0, max_value=200.0, value=70.0)
    family_history_with_overweight = st.selectbox("Riwayat Keluarga Kegemukan", ["yes", "no"])
    FAVC = st.selectbox("Sering Konsumsi Makanan Tinggi Kalori", ["yes", "no"])
    FCVC = st.slider("Frekuensi Konsumsi Sayur (0-3)", 0.0, 3.0, 2.0)
    NCP = st.slider("Jumlah Makan Utama per Hari", 1.0, 4.0, 3.0)
    CAEC = st.selectbox("Makan Antara Waktu Makan", ["no", "Sometimes", "Frequently", "Always"])
    SMOKE = st.selectbox("Merokok", ["yes", "no"])
    CH2O = st.slider("Jumlah Air per Hari", 0.0, 3.0, 2.0)
    SCC = st.selectbox("Konsumsi Makanan Kalori Ekstra", ["yes", "no"])
    FAF = st.slider("Aktivitas Fisik (0-3)", 0.0, 3.0, 1.0)
    TUE = st.slider("Waktu Layar per Hari (0-3)", 0.0, 3.0, 2.0)
    CALC = st.selectbox("Konsumsi Alkohol", ["no", "Sometimes", "Frequently", "Always"])
    MTRANS = st.selectbox("Moda Transportasi", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])

    submitted = st.form_submit_button("Prediksi")

if submitted:
    # Buat dataframe dari input
    input_dict = {
        'Gender': [Gender],
        'Age': [Age],
        'Height': [Height],
        'Weight': [Weight],
        'family_history_with_overweight': [family_history_with_overweight],
        'FAVC': [FAVC],
        'FCVC': [FCVC],
        'NCP': [NCP],
        'CAEC': [CAEC],
        'SMOKE': [SMOKE],
        'CH2O': [CH2O],
        'SCC': [SCC],
        'FAF': [FAF],
        'TUE': [TUE],
        'CALC': [CALC],
        'MTRANS': [MTRANS]
    }
    input_df = pd.DataFrame(input_dict)

    # Preprocessing: get_dummies
    input_df = pd.get_dummies(input_df)

    # === Fix Final Aman ===
    # Tambah kolom yang hilang (harus 0 jika tidak ada)
    missing_cols = [col for col in feature_columns if col not in input_df.columns]
    for col in missing_cols:
        input_df[col] = 0

    # Hapus kolom yang tidak dikenali model
    extra_cols = [col for col in input_df.columns if col not in feature_columns]
    if extra_cols:
        input_df.drop(columns=extra_cols, inplace=True)

    # Urutkan kolom sesuai model
    input_df = input_df[feature_columns]

    # Transformasi dengan scaler
    input_scaled = scaler.transform(input_df)

    # Prediksi
    pred = model.predict(input_scaled)
    label = label_encoder.inverse_transform(pred)

    st.success(f"Hasil Klasifikasi: **{label[0]}**")
