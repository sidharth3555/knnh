import streamlit as st
import joblib
import pandas as pd

# PAGE CONFIG
st.set_page_config(
    page_title="Heart Disease Model prediction using KNN",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# LOAD MODEL AND SCALER
@st.cache_resource(show_spinner=False)
def load_artifacts():
    import os
    BASE_DIR = os.path.dirname(__file__)
    model = joblib.load(os.path.join(BASE_DIR, "knn_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    return model, scaler

knn_model, scaler = load_artifacts()

# STYLING + ANIMATION
st.markdown("""
<style>
body, .block-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: #f9fafb;
    color: #1e293b;
}
h1 {
    font-weight: 800;
    font-size: 3rem;
    color: #ef4444;
    text-align: center;
    animation: fadeInDown 1s ease forwards;
    opacity: 0;
}
.manual-input-container {
    max-width: 900px;
    background: linear-gradient(135deg, #fed7aa 0%, #f97316 100%);
    padding: 30px 40px;
    border-radius: 20px;
    box-shadow: 0 15px 30px rgba(249, 115, 22, 0.4);
    margin: 0 auto 3rem auto;
    animation: fadeInUp 1.5s ease forwards;
}
label {
    font-weight: 700 !important;
    color: #4b5563 !important;
}
div.stButton > button {
    background: #ef4444;
    color: white;
    font-weight: 700;
    font-size: 1.1rem;
    padding: 12px 0;
    border-radius: 12px;
    width: 100%;
    box-shadow: 0 8px 15px rgba(239, 68, 68, 0.4);
}
.result-success, .result-error {
    padding: 15px 20px;
    border-radius: 10px;
    font-weight: 700;
    font-size: 1.2rem;
    margin-top: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
    animation: pulseGlow 2.5s infinite;
}
.result-success {
    background: #d1fae5;
    border-left: 6px solid #10b981;
    color: #065f46;
}
.result-error {
    background: #fee2e2;
    border-left: 6px solid #b91c1c;
    color: #7f1d1d;
}
.result-icon {
    font-size: 1.6rem;
}
@keyframes fadeInDown {
    0% {opacity: 0; transform: translateY(-20px);}
    100% {opacity: 1; transform: translateY(0);}
}
@keyframes fadeInUp {
    0% {opacity: 0; transform: translateY(20px);}
    100% {opacity: 1; transform: translateY(0);}
}
@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 8px 3px rgba(255, 0, 0, 0.2);}
    50% { box-shadow: 0 0 15px 6px rgba(255, 0, 0, 0.6);}
}
</style>
""", unsafe_allow_html=True)

# TITLE
st.title("❤️ Heart Disease Model prediction using KNN")
st.markdown("Enter  values and get real-time prediction.")

# MANUAL INPUT FORM
feature_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

with st.container():
    st.markdown('<div class="manual-input-container">', unsafe_allow_html=True)

    with st.form("manual_input_form"):
        st.subheader("🧮 Input")

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=76)
            sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1], index=0)
            cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3], index=2)
            trestbps = st.number_input("Resting Blood Pressure (trestbps)", value=140)
            chol = st.number_input("Serum Cholestoral (chol)", value=197)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1], index=0)
            restecg = st.selectbox("Resting ECG Results (restecg)", [0, 1, 2], index=1)

        with col2:
            thalach = st.number_input("Max Heart Rate Achieved (thalach)", value=116)
            exang = st.selectbox("Exercise-Induced Angina (exang)", [0, 1], index=0)
            oldpeak = st.number_input("ST Depression (oldpeak)", value=1.10, format="%.2f")
            slope = st.selectbox("Slope of Peak Exercise ST Segment (slope)", [0, 1, 2], index=1)
            ca = st.selectbox("Number of Major Vessels Colored (ca)", [0, 1, 2, 3], index=0)
            thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3], index=1)

        submitted = st.form_submit_button("🔍 Predict")

        if submitted:
            input_data = [[
                age, sex, cp, trestbps, chol, fbs, restecg,
                thalach, exang, oldpeak, slope, ca, thal
            ]]
            try:
                input_df = pd.DataFrame(input_data, columns=feature_names)
                input_scaled = scaler.transform(input_df)
                prediction = knn_model.predict(input_scaled)[0]
                proba = knn_model.predict_proba(input_scaled)[0]
                prob_no_disease = proba[0] * 100
                prob_disease = proba[1] * 100

                if prediction == 1:
                    st.markdown(
                        f'<div class="result-error"><span class="result-icon">🚨</span> '
                        f'**Prediction: Likely Heart Disease** ({prob_disease:.2f}%)</div>',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<div class="result-success"><span class="result-icon">✅</span> '
                        f'**Prediction: No Heart Disease** ({prob_no_disease:.2f}%)</div>',
                        unsafe_allow_html=True)

            except Exception as e:
                st.error(f"⚠️ Prediction error: {e}")

    st.markdown('</div>', unsafe_allow_html=True)
