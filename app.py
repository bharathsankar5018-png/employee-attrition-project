
import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="📊",
    layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "b1gvini/ibm-hr-analytics-attrition-dataset/"
    "master/WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

MODEL_PATH = "employee_attrition_project.pkl"


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        saved_data = pickle.load(file)

    return (
        saved_data["model"],
        saved_data["encoders"],
        saved_data["features"]
    )


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_URL)


try:
    model, encoders, features = load_model()
except Exception as e:
    st.error("Model file could not be loaded.")
    st.code(str(e))
    st.stop()


st.title("AI-Based Employee Attrition Prediction")
st.write("Employee Attrition Prediction and HR Analytics System")

dashboard_tab, prediction_tab, about_tab = st.tabs(
    ["HR Dashboard", "Predict Attrition", "About Project"]
)


# ==========================================
# HR DASHBOARD
# ==========================================

with dashboard_tab:
    st.header("HR Analytics Dashboard")

    try:
        df = load_dataset()

        total_employees = len(df)
        employees_left = (df["Attrition"] == "Yes").sum()
        employees_stayed = (df["Attrition"] == "No").sum()

        attrition_rate = (
            employees_left / total_employees * 100
            if total_employees > 0 else 0
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Employees", total_employees)
        col2.metric("Employees Left", int(employees_left))
        col3.metric("Employees Stayed", int(employees_stayed))
        col4.metric("Attrition Rate", f"{attrition_rate:.1f}%")

        st.subheader("Employee Attrition Overview")

        attrition_count = (
            df["Attrition"]
            .value_counts()
            .reindex(["No", "Yes"], fill_value=0)
        )

        st.bar_chart(attrition_count)

        st.subheader("Overtime vs Attrition")

        overtime = pd.crosstab(
            df["OverTime"],
            df["Attrition"]
        )

        st.bar_chart(overtime)

        st.subheader("Department-wise Attrition Analysis")

        department_data = pd.crosstab(
            df["Department"],
            df["Attrition"]
        )

        st.bar_chart(
            department_data.reindex(
                columns=["No", "Yes"],
                fill_value=0
            )
        )

        st.subheader("Department Attrition Rate (%)")

        department_rate = (
            df.assign(
                Left=df["Attrition"].eq("Yes").astype(int)
            )
            .groupby("Department")["Left"]
            .mean()
            .mul(100)
            .round(2)
        )

        st.bar_chart(department_rate)

        st.subheader("Job Satisfaction vs Attrition")

        satisfaction = pd.crosstab(
            df["JobSatisfaction"],
            df["Attrition"]
        )

        st.bar_chart(satisfaction)

        # FEATURE IMPORTANCE ANALYSIS

        st.subheader("Feature Importance Analysis")

        importance_df = pd.DataFrame({
            "Feature": features,
            "Importance": model.feature_importances_
        })

        importance_df = importance_df.sort_values(
            by="Importance",
            ascending=False
        ).head(10)

        st.write(
            "Top 10 factors used by the model "
            "for employee attrition prediction."
        )

        st.bar_chart(
            importance_df.set_index("Feature")["Importance"]
        )

        with st.expander("View Feature Importance Values"):
            st.dataframe(
                importance_df,
                use_container_width=True
            )

        st.caption(
            "Feature importance indicates how the model uses "
            "features. It does not prove that a feature causes "
            "employee attrition."
        )

        st.caption(
            "Dashboard charts use the IBM HR Analytics sample "
            "dataset, not your company's live employee data."
        )

    except Exception as e:
        st.warning(
            "Dashboard dataset could not be loaded. "
            "Please check the internet connection and try again."
        )
        st.caption(str(e))


# ==========================================
# PREDICT EMPLOYEE ATTRITION
# ==========================================

with prediction_tab:
    st.header("Predict Employee Attrition Risk")
    st.write("Enter employee details below.")

    defaults = {
        "Age": 35,
        "DailyRate": 800,
        "DistanceFromHome": 5,
        "Education": 3,
        "EnvironmentSatisfaction": 3,
        "HourlyRate": 65,
        "JobInvolvement": 3,
        "JobLevel": 2,
        "JobSatisfaction": 3,
        "MonthlyIncome": 5000,
        "MonthlyRate": 14000,
        "NumCompaniesWorked": 2,
        "PercentSalaryHike": 14,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 3,
        "StockOptionLevel": 1,
        "TotalWorkingYears": 10,
        "TrainingTimesLastYear": 3,
        "WorkLifeBalance": 3,
        "YearsAtCompany": 5,
        "YearsInCurrentRole": 3,
        "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 3
    }

    with st.form("employee_form"):
        values = {}

        col1, col2 = st.columns(2)

        for i, feature in enumerate(features):
            container = col1 if i % 2 == 0 else col2

            with container:
                if (
                    feature in encoders
                    and feature != "Attrition"
                ):
                    options = encoders[feature].classes_.tolist()

                    values[feature] = st.selectbox(
                        feature,
                        options
                    )
                else:
                    values[feature] = st.number_input(
                        feature,
                        min_value=0,
                        value=defaults.get(feature, 0),
                        step=1
                    )

        submitted = st.form_submit_button(
            "Predict Attrition Risk"
        )

    if submitted:
        try:
            row = {}

            for feature in features:
                value = values[feature]

                if (
                    feature in encoders
                    and feature != "Attrition"
                ):
                    value = encoders[feature].transform(
                        [value]
                    )[0]

                row[feature] = value

            input_df = pd.DataFrame(
                [row],
                columns=features
            )

            probability = model.predict_proba(input_df)[0][1]

            st.subheader("Prediction Result")

            st.metric(
                "Estimated Attrition Probability",
                f"{probability * 100:.1f}%"
            )

            st.progress(float(probability))

            if probability >= 0.30:
                st.warning("Higher predicted attrition risk")
            else:
                st.success("Lower predicted attrition risk")

            st.caption(
                "This is a model estimate, not a certainty. "
                "Do not use it alone for employment decisions."
            )

        except Exception as e:
            st.error(f"Prediction error: {e}")


# ==========================================
# ABOUT PROJECT
# ==========================================

with about_tab:
    st.header("About This Project")

    st.write(
        "An academic Data Science project that uses a "
        "Random Forest machine learning model to estimate "
        "employee attrition risk."
    )

    st.markdown("""
    **Technologies Used**
    - Python
    - Pandas
    - Scikit-learn
    - Random Forest Classifier
    - Streamlit
    - IBM HR Analytics Sample Dataset

    **Project Objectives**
    - Analyze employee attrition patterns.
    - Compare attrition across departments.
    - Examine overtime and job satisfaction.
    - Display model feature importance.
    - Estimate employee attrition risk using machine learning.
    """)

    st.info(
        "This is an academic demonstration using sample data. "
        "Predictions should not be treated as certain outcomes."
    )
