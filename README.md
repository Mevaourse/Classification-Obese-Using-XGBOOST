
# 📦 Obesity Classification App (XGBoost, RandomForest, GBoost)

[![Streamlit App](https://img.shields.io/badge/Launch%20App-Streamlit-ff4b4b?logo=streamlit&logoColor=white)](https://classification-obese-using-xgboost-hiwzwpduycsd6p7ibttp47.streamlit.app/)

An interactive machine learning web app for predicting obesity levels based on user lifestyle and health data. The app uses **XGBoost**, **Random Forest**, and **Gradient Boosting** classifiers, with hyperparameter tuning and SMOTE balancing.

---

## 📊 Dataset

The app uses the **Obesity Level Estimation dataset**, available on [Kaggle](https://www.kaggle.com/datasets/sid321axn/obesity-level-estimation).

Features include:
- `Gender`, `Age`, `Height`, `Weight`
- Eating habits (`FAVC`, `FCVC`, `NCP`, `CH2O`)
- Physical activity (`FAF`, `TUE`)
- Health indicators and lifestyle (`SMOKE`, `family_history_with_overweight`, etc.)
- Transport method, alcohol, fast food, etc.
- **Target variable:** `NObeyesdad` (obesity category)

---

## 🚀 App Features

- **Landing Page**: Introduction, goal, and model info
- **Prediction Form**: Input health/lifestyle data
- **Model Selection**: XGBoost Tuned, Gradient Boosting Tuned, Random Forest Tuned
- **Preprocessing**: Auto-handled categorical encoding, scaling, and input validation

---

## 📁 Project Structure

```
.
├── app.py                      # Streamlit application
├── models/                     # Trained models and preprocessing tools
│   ├── xgb_tuned.pkl
│   ├── gb_tuned.pkl
│   ├── rf_tuned.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── feature_columns.pkl
│   └── multi_label_encoders.pkl
├── ObesityDataSet.csv          # Original dataset
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 💻 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/Mevaourse/Classification-Obese-Using-XGBOOST-RandomForest-GBoost.git
cd Classification-Obese-Using-XGBOOST-RandomForest-GBoost
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the App
```bash
streamlit run app.py
```

---

## 🧠 Machine Learning Details

- **Models Used**:
  - XGBoost (with tuning)
  - Gradient Boosting
  - Random Forest
- **Preprocessing**:
  - Missing value handling (mean/mode)
  - Label encoding (with per-column encoder saved)
  - Feature scaling (StandardScaler)
  - SMOTE for class balancing
- **Hyperparameter Tuning**:
  - RandomizedSearchCV (10 iterations)

---

## 📈 Model Performance (Test Set)

| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| XGBoost Tuned       | ~96%     | ✅        | ✅     | ✅       |
| Gradient Boosting   | ~95%     | ✅        | ✅     | ✅       |
| Random Forest       | ~94%     | ✅        | ✅     | ✅       |

---

## 🔒 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

Built by [@Mevaourse](https://github.com/Mevaourse).  
Feel free to fork, contribute, or open an issue!

---

### 🌐 Launch the App

> 👉 [Click here to open the live Streamlit app](https://classification-obese-using-xgboost-hiwzwpduycsd6p7ibttp47.streamlit.app/)
