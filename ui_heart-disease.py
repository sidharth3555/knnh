import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os

st.set_page_config(
    page_title="Heart Disease Batch Tester",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("❤️ Heart Disease Model Tester")
st.markdown("""
1. Load the saved KNN model and the saved StandardScaler.
2. Predict on the entire dataset you upload.
3. Display the overall accuracy, confusion matrix, and classification report.
""")

# -------------------------------------------------------------------
# 1) LOAD SAVED MODEL + SCALER
# -------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """
    Load the trained KNN model and the StandardScaler from disk.
    """
    BASE_DIR = os.path.dirname(__file__)

    knn_model_path = os.path.join(BASE_DIR, "knn_model.pkl")
    scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

    try:
        knn_model = joblib.load(knn_model_path)
    except FileNotFoundError:
        st.error(f"❌ Could not find 'knn_model.pkl' at {knn_model_path}")
        knn_model = None

    try:
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        st.error(f"❌ Could not find 'scaler.pkl' at {scaler_path}")
        scaler = None

    return knn_model, scaler

knn_model, scaler = load_artifacts()
if knn_model is None or scaler is None:
    st.stop()

st.markdown("---")

# -------------------------------------------------------------------
# 2) FILE UPLOADER
# -------------------------------------------------------------------
uploaded_file = st.file_uploader(
    label="📂 Upload CSV File for Batch Testing",
    type=["csv"],
    help="""
    The CSV should contain all feature columns in the SAME order/with the SAME names
    that you used during training, plus the `target` column. 
    Example columns: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target
    """
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"⚠️ Unable to read the CSV file: {e}")
        st.stop()

    st.write("**Preview of your uploaded data:**")
    st.dataframe(df.head())

    if "target" not in df.columns:
        st.error("❌ The uploaded CSV does not contain a `target` column.")
        st.stop()

    # 3) SEPARATE FEATURES AND TARGET
    X = df.drop("target", axis=1)
    y = df["target"]

    # 4) SCALE FEATURES
    try:
        X_scaled = scaler.transform(X)
    except Exception as e:
        st.error(f"⚠️ Error while scaling features. Make sure the CSV columns match exactly what the scaler expects. Details:\n{e}")
        st.stop()

    # 5) MAKE PREDICTIONS
    try:
        preds = knn_model.predict(X_scaled)
    except Exception as e:
        st.error(f"⚠️ Error during prediction. Ensure the KNN model and features align. Details:\n{e}")
        st.stop()

    # 6) CALCULATE METRICS
    acc = accuracy_score(y, preds)
    cm = confusion_matrix(y, preds)
    cr = classification_report(y, preds, zero_division=0, output_dict=False)

    # 7) DISPLAY RESULTS
    st.markdown("## 📊 Results")
    st.markdown(f"**Accuracy:** `{acc:.4f}`")

    st.markdown("### Confusion Matrix")
    cm_df = pd.DataFrame(
        cm,
        index=["Actual: No Disease (0)", "Actual: Disease (1)"],
        columns=["Predicted: No Disease (0)", "Predicted: Disease (1)"]
    )
    st.table(cm_df)

    st.markdown("### Classification Report")
    st.text(cr)

    st.success("✅ Batch evaluation complete!")

    # OPTIONAL: Show the raw predictions side by side
    if st.checkbox("Show predictions alongside uploaded data"):
        results_df = df.copy()
        results_df["predicted_target"] = preds
        st.subheader("Uploaded Data + Predictions")
        st.dataframe(results_df)
else:
    st.info("ℹ️ Please upload a CSV file to get started.")
