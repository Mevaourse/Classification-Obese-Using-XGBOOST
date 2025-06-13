import streamlit as st
import pandas as pd
import joblib

# ===== Load models and tools =====
xgb_model = joblib.load('models/xgb_tuned.pkl')
gb_model = joblib.load('models/gb_tuned.pkl')
rf_model = joblib.load('models/rf_tuned.pkl')
scaler = joblib.load('models/scaler.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')
feature_columns = joblib.load('models/feature_columns.pkl')

# ===== Sidebar Navigation =====
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Beranda", "Prediksi"])

# ===== Landing Page / Home =====
if page == "Beranda":
    st.title("📊 Aplikasi Klasifikasi Obesitas")
    st.markdown("""
    ### Tentang Aplikasi
    Aplikasi ini dibuat untuk memprediksi **kategori obesitas** seseorang berdasarkan data kesehatan dan gaya hidup seperti berat badan, tinggi badan, kebiasaan olahraga, pola makan, dan lainnya.

    ### Tujuan
    - Membantu pengguna mengetahui potensi risiko obesitas secara cepat.
    - Mempromosikan kesadaran tentang pentingnya pola hidup sehat.
    - Menerapkan model machine learning terbaik untuk prediksi.

    ### Model yang Digunakan
    1. **XGBoost (Tuned)** - Akurasi tinggi dengan boosting decision tree.
    2. **Gradient Boosting (Tuned)** - Model yang kuat untuk menangani kompleksitas.
    3. **Random Forest (Tuned)** - Model yang stabil dan kuat untuk klasifikasi umum.

    ### Evaluasi Akurasi Model (Setelah Tuning)
    | Model | Akurasi |
    |-------|---------|
    | XGBoost Tuned | ✅ ~96% |
    | Gradient Boosting Tuned | ✅ ~95% |
    | Random Forest Tuned | ✅ ~94% |

    Silakan buka halaman *Prediksi* untuk mencoba aplikasi ini secara langsung.
    """)

# ===== Prediction Page =====
elif page == "Prediksi":
    st.title("🔍 Prediksi Kategori Obesitas")

    # Pilihan model
    model_choice = st.selectbox("Pilih Model", ["XGBoost Tuned", "Gradient Boosting Tuned", "Random Forest Tuned"])

    # Input fitur
    def user_input_features():
        data = {}
        for col in feature_columns:
            if col in ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']:
                data[col] = st.selectbox(col, [0, 1])  # sesuaikan jika ingin lebih user-friendly
            else:
                data[col] = st.number_input(col, step=0.1)
        return pd.DataFrame([data])

    input_df = user_input_features()

    if st.button("Prediksi"):
        input_df = input_df[feature_columns]
        input_scaled = scaler.transform(input_df)

        # Pilih model
        if model_choice == "XGBoost Tuned":
            model = xgb_model
        elif model_choice == "Gradient Boosting Tuned":
            model = gb_model
        else:
            model = rf_model

        # Prediksi
        pred = model.predict(input_scaled)
        label = label_encoder.inverse_transform(pred)

        st.success(f"Prediksi Kategori Obesitas: **{label[0]}**")
