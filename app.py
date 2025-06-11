import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Fungsi untuk memuat model dan scaler
@st.cache_resource
def load_resources():
    scaler = joblib.load('models/scaler.pkl')
    model = joblib.load('models/xgb_best.pkl')
    feature_columns = joblib.load('models/feature_columns.pkl')
    label_encoder = joblib.load('models/label_encoder.pkl')
    return scaler, model, feature_columns , label_encoder

scaler, model, feature_columns = load_resources()

# Judul Aplikasi
st.title("Aplikasi Klasifikasi Obesitas")

st.write("""
Aplikasi ini memprediksi tingkat obesitas berdasarkan beberapa fitur.
""")

# Bagian Input Pengguna
st.header("Input Data Pengguna")

# Buat input untuk setiap fitur
gender = st.selectbox("Jenis Kelamin", ['Male', 'Female'])
age = st.slider("Umur", 0, 100, 25)
height = st.number_input("Tinggi Badan (cm)", min_value=50.0, max_value=300.0, value=170.0)
weight = st.number_input("Berat Badan (kg)", min_value=10.0, max_value=500.0, value=60.0)
family_history_with_overweight = st.selectbox("Riwayat Keluarga dengan Obesitas", ['yes', 'no'])
FAVC = st.selectbox("Konsumsi Makanan Tinggi Kalori", ['yes', 'no'])
FCVC = st.slider("Konsumsi Sayuran (kali/hari)", 1.0, 3.0, 2.0)
NCP = st.slider("Jumlah Makanan Utama (kali/hari)", 1.0, 4.0, 3.0)
CAEC = st.selectbox("Konsumsi Makanan di Antara Waktu Makan", ['no', 'Sometimes', 'Frequently', 'Always'])
SMOKE = st.selectbox("Merokok", ['yes', 'no'])
SCC = st.selectbox("Monitoring Asupan Kalori", ['yes', 'no'])
CH2O = st.slider("Konsumsi Air (liter/hari)", 1.0, 3.0, 2.0)
FAF = st.slider("Aktivitas Fisik (kali/minggu)", 0.0, 3.0, 1.0)
TUE = st.slider("Waktu Penggunaan Perangkat (jam/hari)", 0.0, 2.0, 0.0)
CALC = st.selectbox("Konsumsi Alkohol", ['no', 'Sometimes', 'Frequently', 'Always'])
MTRANS = st.selectbox("Alat Transportasi yang Digunakan", ['Public_Transportation', 'Walking', 'Automobile', 'Motorbike', 'Bike'])

# Konversi input kategori ke numerik sesuai dengan cara pelatihan model
# Anda perlu tahu bagaimana setiap kategori dikonversi selama pelatihan.
# Ini adalah contoh sederhana, Anda mungkin perlu menyesuaikannya dengan LabelEncoder yang digunakan.
input_data = {
    'Gender': 1 if gender == 'Male' else 0, # Contoh konversi binary
    'Age': age,
    'Height': height,
    'Weight': weight,
    'family_history_with_overweight': 1 if family_history_with_overweight == 'yes' else 0, # Contoh konversi binary
    'FAVC': 1 if FAVC == 'yes' else 0, # Contoh konversi binary
    'FCVC': FCVC,
    'NCP': NCP,
    'CAEC': {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}[CAEC], # Contoh konversi multiple category
    'SMOKE': 1 if SMOKE == 'yes' else 0, # Contoh konversi binary
    'SCC': 1 if SCC == 'yes' else 0, # Contoh konversi binary
    'CH2O': CH2O,
    'FAF': FAF,
    'TUE': TUE,
    'CALC': {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}[CALC], # Contoh konversi multiple category
    'MTRANS': {'Public_Transportation': 0, 'Walking': 1, 'Automobile': 2, 'Motorbike': 3, 'Bike': 4}[MTRANS] # Contoh konversi multiple category
}

# Buat DataFrame dari input pengguna
input_df = pd.DataFrame([input_data])

# Pastikan urutan kolom sesuai dengan feature_columns
input_df = input_df[feature_columns]

# Lakukan scaling pada input pengguna
input_scaled = scaler.transform(input_df)

# Tombol Prediksi
if st.button("Prediksi Tingkat Obesitas"):
    # Lakukan prediksi
    prediction_numeric = model.predict(input_scaled)

    # Konversi hasil prediksi numerik kembali ke label kategori asli
    # Ini membutuhkan LabelEncoder yang digunakan saat pelatihan
    # Jika Anda tidak menyimpannya, Anda mungkin perlu mendefinisikan mapping manual
    # atau melatih ulang LabelEncoder di sini dengan data asli.
    # Contoh konversi manual (sesuaikan dengan data Anda)
    label_mapping = {
        0: 'Insufficient_Weight',
        1: 'Normal_Weight',
        2: 'Overweight_Level_I',
        3: 'Overweight_Level_II',
        4: 'Obesity_Type_I',
        5: 'Obesity_Type_II',
        6: 'Obesity_Type_III'
    }
    predicted_label = label_mapping.get(prediction_numeric[0], "Unknown") # Menggunakan get untuk menghindari error jika label tidak ditemukan

    st.subheader("Hasil Prediksi")
    st.write(f"Tingkat Obesitas yang Diprediksi: **{predicted_label}**")
