
import pickle
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Employee HR Analytics",
    page_icon="📊",
    layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "b1gvini/ibm-hr-analytics-attrition-dataset/"
    "master/WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

@st.cache_resource
def load_model():
    with open("employee_attrition_project.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_URL)

st.title("📊 Employee Attrition Prediction & HR Analytics")
st.caption("Machine Learning | Random Forest | HR Analytics")

try:
    project = load_model()
    model = project["model"]
    encoders = project["encoders"]
    features = project["features"]
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()

dashboard_tab, prediction_tab, about_tab = st.tabs(
    ["📈 HR Dashboard", "🔮 Predict Attrition", "ℹ️ About Project"]
)

with dashboard_tab:
    st.header("HR Analytics Dashboard")

    try:
        df = load_dataset()

        total = len(df)
        left_count = (df["Attrition"] == "Yes").sum()
        stayed_count = (df["Attrition"] == "No").sum()
        attrition_rate = left_count / total * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Employees", f"{total:,}")
        c2.metric("Employees Left", f"{left_count:,}")
        c3.metric("Employees Stayed", f"{stayed_count:,}")
        c4.metric("Attrition Rate", f"{attrition_rate:.1f}%")

        st.divider()

        left, right = st.columns(2)

        with left:
            st.subheader("Employee Attrition")
            attrition_counts = (
                df["Attrition"].value_counts()
                .reindex(["No", "Yes"], fill_value=0)
            )
            st.bar_chart(attrition_counts)

        with right:
            st.subheader("Overtime vs Attrition")
            
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

            overtime = pd.crosstab(
                df["OverTime"], df["Attrition"]
            )
            st.bar_chart(overtime)

        st.subheader("Job Satisfaction vs Attrition")
        satisfaction = pd.crosstab(
            df["JobSatisfaction"], df["Attrition"]
        )
        st.bar_chart(satisfaction)

        st.caption(
            "Dashboard charts use the IBM HR Analytics sample dataset. "
            "This is sample data, not your company's live employee data."
        )

    except Exception as e:
        st.warning(
            "Dashboard dataset could not be loaded. "
            "Please check the internet connection and try again."
        )
        st.caption(str(e))

with prediction_tab:
    st.header("Predict Employee Attrition Risk")
    st.write("Enter employee details below.")

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

        col1, col2 = st.columns(2)

        for i, feature in enumerate(features):
            container = col1 if i % 2 == 0 else col2

            with container:
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
        try:
            row = {}

            for feature in features:
                value = values[feature]

                if feature in encoders and feature != "Attrition":
                    value = encoders[feature].transform([value])[0]

                row[feature] = value

            input_df = pd.DataFrame([row], columns=features)
            probability = model.predict_proba(input_df)[0][1]

            st.subheader("Prediction Result")
            st.metric(
                "Estimated Attrition Probability",
                f"{probability * 100:.1f}%"
            )

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

with about_tab:
    st.header("About This Project")
    st.write(
        "An academic Data Science project that uses a Random Forest "
        "machine learning model to estimate employee attrition risk."
    )
    st.markdown("""
    **Technologies used**
    - Python
    - Pandas
    - Scikit-learn
    - Random Forest Classifier
    - Streamlit
    - IBM HR Analytics sample dataset
    """)
