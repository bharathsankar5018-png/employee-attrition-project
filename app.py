
import pickle
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="📊"
)

@st.cache_resource
def load_model():
    with open("employee_attrition_project.pkl", "rb") as f:
        return pickle.load(f)

st.title("Employee Attrition Prediction System")
st.write("Predict employee attrition risk using machine learning.")

try:
    project = load_model()
    model = project["model"]
    encoders = project["encoders"]
    features = project["features"]

    defaults = {
        "Age": 35, "DailyRate": 800, "DistanceFromHome": 5,
        "Education": 3, "EnvironmentSatisfaction": 3,
        "HourlyRate": 65, "JobInvolvement": 3, "JobLevel": 2,
        "JobSatisfaction": 3, "MonthlyIncome": 5000,
        "MonthlyRate": 14000, "NumCompaniesWorked": 2,
        "PercentSalaryHike": 14, "PerformanceRating": 3,
        "RelationshipSatisfaction": 3, "StockOptionLevel": 1,
        "TotalWorkingYears": 10, "TrainingTimesLastYear": 3,
        "WorkLifeBalance": 3, "YearsAtCompany": 5,
        "YearsInCurrentRole": 3, "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 3
    }

    with st.form("employee_form"):
        values = {}

        for i, feature in enumerate(features):
            if feature in encoders and feature != "Attrition":
                options = encoders[feature].classes_.tolist()
                values[feature] = st.selectbox(feature, options)
            else:
                values[feature] = st.number_input(
                    feature,
                    min_value=0,
                    value=defaults.get(feature, 0),
                    step=1
                )

        submitted = st.form_submit_button("Predict Attrition Risk")

    if submitted:
        row = {}

        for feature in features:
            value = values[feature]
            if feature in encoders and feature != "Attrition":
                value = encoders[feature].transform([value])[0]
            row[feature] = value

        input_df = pd.DataFrame([row], columns=features)
        probability = model.predict_proba(input_df)[0][1]

        st.metric("Estimated Attrition Probability", f"{probability:.1%}")

        if probability >= 0.30:
            st.warning("Higher predicted attrition risk")
        else:
            st.success("Lower predicted attrition risk")

        st.caption(
            "This is an educational model estimate, not a certainty. "
            "Do not use it alone to make employment decisions."
        )

except FileNotFoundError:
    st.info(
        "Model file is not uploaded yet. Upload "
        "employee_attrition_project.pkl to this repository."
    )
except Exception as e:
    st.error(f"Error: {e}")
