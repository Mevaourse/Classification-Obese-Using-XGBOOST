import streamlit as st
import pandas as pd
import joblib

# Load models and objects
xgb_model = joblib.load('models/xgb_tuned.pkl')
gb_model = joblib.load('models/gb_tuned.pkl')
rf_model = joblib.load('models/rf_tuned.pkl')
scaler = joblib.load('models/scaler.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')
feature_columns = joblib.load('models/feature_columns.pkl')

# UI Title
st.title("Aplikasi Klasifikasi Obesitas")

# Sidebar untuk memilih model
model_choice = st.sidebar.selectbox("Pilih Model", ["XGBoost Tuned", "Gradient Boosting Tuned", "Random Forest Tuned"])

# Input user
def user_input():
    data = {}
    for col in feature_columns:
        if col in ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']:
            data[col] = st.selectbox(f"{col}", options=[0, 1])  # atau sesuaikan label encoder
        else:
            data[col] = st.number_input(f"{col}", step=0.1)
    return pd.DataFrame([data])

input_df = user_input()

if st.button("Prediksi"):
    # Pastikan urutan kolom benar
    input_df = input_df[feature_columns]
    
    # Scaling
    input_scaled = scaler.transform(input_df)
    
    # Model prediksi
    if model_choice == "XGBoost Tuned":
        model = xgb_model
    elif model_choice == "Gradient Boosting Tuned":
        model = gb_model
    else:
        model = rf_model

    pred = model.predict(input_scaled)
    label = label_encoder.inverse_transform(pred)
    
    st.success(f"Prediksi Kategori Obesitas: {label[0]}")
