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

# STYLING + ANIMATION + DOCTOR CARD CSS
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

/* Doctor Cards */
.doctor-card {
    background: white;
    border-radius: 15px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    padding: 15px 20px;
    margin-bottom: 20px;
}
.doctor-name {
    font-size: 1.3rem;
    font-weight: 700;
    color: #ef4444;
    margin-bottom: 8px;
}
.doctor-info {
    font-size: 1rem;
    margin-bottom: 6px;
    color: #334155;
}
iframe {
    border-radius: 12px;
    margin-top: 10px;
    border: none;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# TITLE
st.title("❤ Heart Disease Model prediction using KNN")
st.markdown("Enter values and get real-time prediction.")

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
                        f'*Prediction: Likely Heart Disease* ({prob_disease:.2f}%)</div>',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<div class="result-success"><span class="result-icon">✅</span> '
                        f'*Prediction: No Heart Disease* ({prob_no_disease:.2f}%)</div>',
                        unsafe_allow_html=True)

            except Exception as e:
                st.error(f"⚠ Prediction error: {e}")

    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------
# DOCTOR INFO DATA
doctors_by_state = {
    "Maharashtra": [
        {"name": "Dr. A. Joshi", "contact": "+91 9876543210", "qualification": "MD, Cardiologist", "map_url": "https://www.google.com/maps?q=18.5204,73.8567&output=embed"},
        {"name": "Dr. M. Deshmukh", "contact": "+91 9876543211", "qualification": "DM, Cardiologist", "map_url": "https://www.google.com/maps?q=19.0760,72.8777&output=embed"},
        {"name": "Dr. R. Kulkarni", "contact": "+91 9876543212", "qualification": "MD, Cardiologist", "map_url": "https://www.google.com/maps?q=21.1458,79.0882&output=embed"},
        {"name": "Dr. S. Gokhale", "contact": "+91 9876543213", "qualification": "DM, Cardiologist", "map_url": "https://www.google.com/maps?q=19.8762,75.3433&output=embed"},
        {"name": "Dr. P. Kamat", "contact": "+91 9876543214", "qualification": "MD, Cardiologist", "map_url": "https://www.google.com/maps?q=20.0111,73.7903&output=embed"},
    ],
    "Karnataka": [
        {"name": "Dr. V. Rao", "contact": "+91 9123456789", "qualification": "MD, Cardiologist", "map_url": "https://www.google.com/maps?q=12.9716,77.5946&output=embed"},
        {"name": "Dr. N. Kumar", "contact": "+91 9123456790", "qualification": "DM, Cardiologist", "map_url": "https://www.google.com/maps?q=15.3173,75.7139&output=embed"},
        {"name": "Dr. S. Shetty", "contact": "+91 9123456791", "qualification": "MD, Cardiologist", "map_url": "https://www.google.com/maps?q=13.1986,77.7066&output=embed"},
        {"name": "Dr. R. Patil", "contact": "+91 9123456792", "qualification": "DM, Cardiologist", "map_url": "https://www.google.com/maps?q=15.9129,74.8410&output=embed"},
        {"name": "Dr. M. Fernandes", "contact": "+91 9123456793", "qualification": "MD, Cardiologist", "map_url": "https://www.google.com/maps?q=12.2958,76.6394&output=embed"},
    ],
    "Tamil Nadu": [
        {"name": "Dr. K. Ramesh", "contact": "+91 9988776655", "qualification": "MD, Cardiologist", "map_url": "https://www.google.com/maps?q=13.0827,80.2707&output=embed"},
        {"name": "Dr. S. Balaji", "contact": "+91 9988776656", "qualification": "DM, Cardiologist", "map_url": "https://www.google.com/maps?q=11.0168,76.9558&output=embed"},
        {"name": "Dr. L. Kumar", "contact": "+91 9988776657", "qualification": "MD, Cardiologist", "map_url": "https://www.google.com/maps?q=10.7905,78.7047&output=embed"},
        {"name": "Dr. P. Mani", "contact": "+91 9988776658", "qualification": "DM, Cardiologist", "map_url": "https://www.google.com/maps?q=9.9252,78.1198&output=embed"},
        {"name": "Dr. M. Sekar", "contact": "+91 9988776659", "qualification": "MD, Cardiologist", "map_url": "https://www.google.com/maps?q=12.2958,76.6394&output=embed"},
    ],
    "Delhi": [
        {"name": "Dr. A. Singh", "contact": "+91 9876541234", "qualification": "MD, Cardiologist", "map_url": "https://www.google.com/maps?q=28.6139,77.2090&output=embed"},
        {"name": "Dr. N. Sharma", "contact": "+91 9876541235", "qualification": "DM, Cardiologist", "map_url": "https://www.google.com/maps?q=28.7041,77.1025&output=embed"},
        {"name": "Dr. R. Gupta", "contact": "+91 9876541236", "qualification": "MD, Cardiologist", "map_url": "https://www.google.com/maps?q=28.5355,77.3910&output=embed"},
        {"name": "Dr. S. Verma", "contact": "+91 9876541237", "qualification": "DM, Cardiologist", "map_url": "https://www.google.com/maps?q=28.4595,77.0266&output=embed"},
        {"name": "Dr. P. Mehta", "contact": "+91 9876541238", "qualification": "MD, Cardiologist", "map_url": "https://www.google.com/maps?q=28.4089,77.3178&output=embed"},
    ],
    "West Bengal": [
        {"name": "Dr. R. Chatterjee", "contact": "+91 9123459876", "qualification": "MD, Cardiologist", "map_url": "https://www.google.com/maps?q=22.5726,88.3639&output=embed"},
        {"name": "Dr. S. Das", "contact": "+91 9123459877", "qualification": "DM, Cardiologist", "map_url": "https://www.google.com/maps?q=23.8103,87.5234&output=embed"},
        {"name": "Dr. L. Mukherjee", "contact": "+91 9123459878", "qualification": "MD, Cardiologist", "map_url": "https://www.google.com/maps?q=22.9786,87.7478&output=embed"},
        {"name": "Dr. M. Sen", "contact": "+91 9123459879", "qualification": "DM, Cardiologist", "map_url": "https://www.google.com/maps?q=24.4539,87.3119&output=embed"},
        {"name": "Dr. T. Bhattacharya", "contact": "+91 9123459880", "qualification": "MD, Cardiologist", "map_url": "https://www.google.com/maps?q=22.8456,88.3621&output=embed"},
    ],
}

# DOCTOR INFORMATION SECTION
st.markdown("---")
st.header("Find Cardiologists Near You")

state = st.selectbox("Select Your State", options=list(doctors_by_state.keys()))

if state:
    st.markdown(f"### Cardiologists in {state}")
    doctors = doctors_by_state[state]

    for doc in doctors:
        st.markdown(f'''
        <div class="doctor-card">
            <div class="doctor-name">{doc["name"]}</div>
            <div class="doctor-info"><strong>Contact:</strong> {doc["contact"]}</div>
            <div class="doctor-info"><strong>Qualification:</strong> {doc["qualification"]}</div>
            <iframe src="{doc["map_url"]}" width="100%" height="200" loading="lazy" allowfullscreen></iframe>
        </div>
        ''', unsafe_allow_html=True)