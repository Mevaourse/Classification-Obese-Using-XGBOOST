import streamlit as st
import pandas as pd
import joblib

# Load model dan objek preprocessing
xgb_model = joblib.load('models/xgb_tuned.pkl')
gb_model = joblib.load('models/gb_tuned.pkl')
rf_model = joblib.load('models/rf_tuned.pkl')
scaler = joblib.load('models/scaler.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')
feature_columns = joblib.load('models/feature_columns.pkl')

# Sidebar Navigation
st.sidebar.title("📌 Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Beranda", "Prediksi"])

# =======================
# Halaman Landing / Beranda
# =======================
if page == "Beranda":
    st.title("📊 Aplikasi Klasifikasi Obesitas")

    st.markdown("""
    ### 📝 Tentang Aplikasi
    Aplikasi ini menggunakan machine learning untuk memprediksi **kategori obesitas** seseorang berdasarkan informasi gaya hidup dan kesehatan.

    ### 🎯 Tujuan
    - Memberikan prediksi cepat dan akurat terkait obesitas.
    - Mendorong kesadaran terhadap gaya hidup sehat.
    - Menerapkan model pembelajaran mesin terbaik untuk klasifikasi.

    ### 🧠 Model yang Digunakan
    - **XGBoost (Tuned)** – Akurasi ~96%
    - **Gradient Boosting (Tuned)** – Akurasi ~95%
    - **Random Forest (Tuned)** – Akurasi ~94%

    --- 
    Pilih halaman **Prediksi** di sidebar untuk mencoba aplikasi ini.
    """)

# =======================
# Halaman Prediksi
# =======================
elif page == "Prediksi":
    st.title("🔍 Prediksi Kategori Obesitas")

    model_choice = st.selectbox("Pilih Model Machine Learning", [
        "XGBoost Tuned", "Gradient Boosting Tuned", "Random Forest Tuned"
    ])

    def input_user():
        data = {}
        for col in feature_columns:
            if col in ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']:
                data[col] = st.selectbox(f"{col}", [0, 1])  # bisa diganti dengan label asli
            else:
                data[col] = st.number_input(f"{col}", step=0.1)
        return pd.DataFrame([data])

    input_df = input_user()

    if st.button("Lakukan Prediksi"):
        input_df = input_df[feature_columns]
        input_scaled = scaler.transform(input_df)

        # Pemilihan model
        if model_choice == "XGBoost Tuned":
            model = xgb_model
        elif model_choice == "Gradient Boosting Tuned":
            model = gb_model
        else:
            model = rf_model

        # Prediksi
        pred = model.predict(input_scaled)
        label = label_encoder.inverse_transform(pred)

        st.success(f"Hasil Prediksi: **{label[0]}**")
