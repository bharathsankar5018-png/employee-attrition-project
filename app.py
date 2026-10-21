
import os
import pickle
import streamlit as st
import pandas as pd

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

# Find the model file in the same folder as app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR,
    "employee_attrition_project.pkl"
)


@st.cache_resource
def load_model():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            "employee_attrition_project.pkl was not found "
            "in the same folder as app.py. Check the GitHub "
            "repository and file name."
        )

    with open(MODEL_PATH, "rb") as file:
        saved_data = pickle.load(file)

    return (
        saved_data["model"],
        saved_data["encoders"],
        saved_data["features"],
        saved_data.get("evaluation")
    )


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_URL)


try:
    model, encoders, features, evaluation = load_model()
except Exception as e:
    st.error("Model file could not be loaded.")
    st.code(str(e))
    st.stop()


st.title("AI-Based Employee Attrition Prediction")
st.write(
    "Employee Attrition Prediction and HR Analytics System"
)

dashboard_tab, prediction_tab, evaluation_tab, about_tab = st.tabs(
    [
        "HR Dashboard",
        "Predict Attrition",
        "Model Evaluation",
        "About Project"
    ]
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

        st.subheader("Feature Importance Analysis")

        if hasattr(model, "feature_importances_"):
            importance_df = pd.DataFrame({
                "Feature": features,
                "Importance": model.feature_importances_
            })

            importance_df = importance_df.sort_values(
                by="Importance",
                ascending=False
            ).head(10)

            st.write(
                "Top 10 features used by the prediction model."
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
                "Feature importance does not prove cause and effect."
            )

        st.caption(
            "Dashboard uses the IBM HR Analytics sample dataset, "
            "not your company's live employee data."
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

            probabilities = model.predict_proba(input_df)[0]

            # Find the probability corresponding to Attrition = Yes
            classes = list(model.classes_)
            yes_index = classes.index(1) if 1 in classes else classes.index("Yes")
            probability = float(probabilities[yes_index])

            st.subheader("Prediction Result")

            st.metric(
                "Estimated Attrition Probability",
                f"{probability * 100:.1f}%"
            )

            st.progress(
                min(max(probability, 0.0), 1.0)
            )

            if probability >= 0.60:
                st.error("HIGH RISK")
                st.write(
                    "The model estimates a higher attrition risk "
                    "for this employee."
                )
            elif probability >= 0.30:
                st.warning("MEDIUM RISK")
                st.write(
                    "The model estimates a moderate attrition risk "
                    "for this employee."
                )
            else:
                st.success("LOW RISK")
                st.write(
                    "The model estimates a lower attrition risk "
                    "for this employee."
                )

            st.caption(
                "Demonstration thresholds: Low below 30%, Medium "
                "from 30% to below 60%, and High at 60% or above."
            )

            st.info(
                "This is a model estimate, not a certainty. "
                "These thresholds are illustrative and have not "
                "been validated for real HR decisions. "
                "Do not use this result alone to make employment decisions."
            )

        except Exception as e:
            st.error(f"Prediction error: {e}")


# ==========================================
# MODEL EVALUATION
# ==========================================

with evaluation_tab:
    st.header("Model Evaluation Dashboard")

    if evaluation is None:
        st.warning(
            "Evaluation metrics were not found in the model file. "
            "Upload the updated employee_attrition_project.pkl "
            "containing the saved evaluation results."
        )
    else:
        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Accuracy",
            f"{evaluation['accuracy'] * 100:.2f}%"
        )
        col2.metric(
            "Precision",
            f"{evaluation['precision'] * 100:.2f}%"
        )
        col3.metric(
            "Recall",
            f"{evaluation['recall'] * 100:.2f}%"
        )
        col4.metric(
            "F1 Score",
            f"{evaluation['f1_score'] * 100:.2f}%"
        )

        st.divider()
        st.subheader("Confusion Matrix")

        cm = evaluation.get("confusion_matrix")

        if cm is not None:
            confusion_df = pd.DataFrame(
                cm,
                index=["Actual: Stayed (No)", "Actual: Left (Yes)"],
                columns=[
                    "Predicted: Stayed (No)",
                    "Predicted: Left (Yes)"
                ]
            )

            st.dataframe(
                confusion_df,
                use_container_width=True
            )

            st.caption(
                "Rows are actual outcomes; columns are predicted "
                "outcomes. The diagonal cells are correct predictions."
            )

        if "threshold" in evaluation:
            st.info(
                f"Evaluation probability threshold: "
                f"{evaluation['threshold']:.2f}"
            )

        st.subheader("Metric Definitions")
        st.markdown("""
        - **Accuracy:** Overall proportion of correct predictions.
        - **Precision:** Among employees predicted to leave, how many actually left.
        - **Recall:** Among employees who actually left, how many were identified.
        - **F1 Score:** Combined measure of precision and recall.
        """)

        st.caption(
            "These metrics describe the saved test-set results. "
            "Performance may differ on new data. This academic "
            "evaluation does not validate the model for real HR decisions."
        )


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
    - Display feature importance.
    - Estimate employee attrition risk.
    - Evaluate the model using Accuracy, Precision, Recall,
      F1 Score, and Confusion Matrix.
    """)

    st.info(
        "This academic project uses sample data. "
        "Predictions are estimates and are not guaranteed outcomes."
    )
