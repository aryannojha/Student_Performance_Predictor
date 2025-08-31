
import streamlit as st
import joblib
import pandas as pd
from .schemas import Features
from .utils import ARTIFACTS_DIR

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")

st.title("🎓 Student Performance Predictor")
st.write("Fill the form and get a predicted final grade (0–100).")

MODEL_PATH = ARTIFACTS_DIR / "model.joblib"

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

with st.form("predict_form"):
    hours_studied = st.number_input("Hours Studied per Day", 0.0, 24.0, 3.0, 0.5)
    attendance_rate = st.slider("Attendance Rate", 0.0, 1.0, 0.9, 0.01)
    past_grades_avg = st.slider("Past Grades Average", 0, 100, 75, 1)
    extracurricular = st.selectbox("Extracurricular", ["yes", "no"])
    parental_involvement = st.selectbox("Parental Involvement", ["low", "medium", "high"])
    sleep_hours = st.number_input("Sleep Hours", 0.0, 24.0, 7.0, 0.5)
    internet_access = st.selectbox("Internet Access", ["yes", "no"])
    test_prep_course = st.selectbox("Test Prep Course", ["none", "completed", "ongoing"])
    gender = st.selectbox("Gender", ["M", "F"])
    age = st.slider("Age", 5, 100, 16, 1)
    school_support = st.selectbox("School Support", ["yes", "no"])
    absences = st.number_input("Absences", 0, 365, 2, 1)

    submitted = st.form_submit_button("Predict")

if submitted:
    if model is None:
        st.error("Model not loaded. Please run training first.")
    else:
        data = Features(
            hours_studied=hours_studied,
            attendance_rate=attendance_rate,
            past_grades_avg=past_grades_avg,
            extracurricular=extracurricular,
            parental_involvement=parental_involvement,
            sleep_hours=sleep_hours,
            internet_access=internet_access,
            test_prep_course=test_prep_course,
            gender=gender,
            age=age,
            school_support=school_support,
            absences=absences
        ).dict()
        df = pd.DataFrame([data])
        pred = float(model.predict(df)[0])
        st.success(f"Predicted Final Grade: **{pred:.1f}** / 100")
