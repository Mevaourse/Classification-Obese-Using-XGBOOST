import streamlit as st
import pandas as pd
import joblib

# === Load Models and Tools ===
xgb_model = joblib.load('models/xgb_tuned.pkl')
gb_model = joblib.load('models/gb_tuned.pkl')
rf_model = joblib.load('models/rf_tuned.pkl')
scaler = joblib.load('models/scaler.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')
feature_columns = joblib.load('models/feature_columns.pkl')
multi_label_encoders = joblib.load('models/multi_label_encoders.pkl')

# === Sidebar Navigation ===
st.sidebar.title("📌 Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Beranda", "Prediksi"])

# === Landing Page ===
if page == "Beranda":
    st.title("📊 Aplikasi Klasifikasi Obesitas")
    st.markdown("""
    ### 📝 Tentang Aplikasi
    Aplikasi ini memanfaatkan machine learning untuk mengklasifikasikan kategori obesitas seseorang berdasarkan data gaya hidup dan kesehatan seperti:
    - Pola makan (jumlah makan, konsumsi air, fast food, alkohol)
    - Aktivitas fisik (frekuensi olahraga, waktu layar)
    - Riwayat keluarga, kebiasaan merokok, transportasi, dan lainnya.

    ### 🎯 Tujuan
    - Menyediakan prediksi cepat untuk risiko obesitas
    - Membantu pengguna memahami faktor yang memengaruhi berat badan

    ### 🤖 Model yang Digunakan
    - **XGBoost Tuned** (~96% akurasi)
    - **Gradient Boosting Tuned** (~95% akurasi)
    - **Random Forest Tuned** (~94% akurasi)

    ---
    Silakan buka menu **Prediksi** di sebelah kiri untuk mencoba aplikasi.
    """)

# === Prediction Page ===
elif page == "Prediksi":
    st.title("🔍 Prediksi Kategori Obesitas")

    # Dokumentasi input
    with st.expander("📘 Petunjuk Pengisian Data"):
        st.markdown("""
        - Masukkan nilai **angka** sesuai kondisi Anda:
            - Contoh: `Weight = 70`, `Height = 1.75`
        - Kolom seperti `Gender`, `MTRANS`, `CAEC`, `CALC` akan muncul dengan label asli, bukan angka.
        - Kolom seperti `family_history_with_overweight`, `FAVC`, `SMOKE`, `SCC` cukup diisi 0 = Tidak, 1 = Ya
        """)

    # Fungsi Input
    def input_user():
        data = {}
        for col in feature_columns:
            if col in multi_label_encoders:
                le = multi_label_encoders[col]
                selected = st.selectbox(f"{col}", le.classes_)
                data[col] = le.transform([selected])[0]
            elif col in ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']:
                data[col] = st.selectbox(f"{col} (0=Tidak, 1=Ya)", [0, 1])
            else:
                data[col] = st.number_input(f"{col}", min_value=0.0, step=0.1)
        return pd.DataFrame([data])

    input_df = input_user()

    # Pilih model
    model_choice = st.selectbox("🧠 Pilih Model Machine Learning", [
        "XGBoost Tuned", "Gradient Boosting Tuned", "Random Forest Tuned"
    ])

    # Prediksi
    if st.button("🔮 Lakukan Prediksi"):
        try:
            input_scaled = scaler.transform(input_df[feature_columns])
            if model_choice == "XGBoost Tuned":
                model = xgb_model
            elif model_choice == "Gradient Boosting Tuned":
                model = gb_model
            else:
                model = rf_model

            pred = model.predict(input_scaled)
            label = label_encoder.inverse_transform(pred)
            st.success(f"✅ Hasil Prediksi: **{label[0]}**")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat prediksi: {e}")
