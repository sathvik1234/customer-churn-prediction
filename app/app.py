
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI-Powered Customer Churn Prediction")
st.markdown(
    """
    Predict customer churn probability using a trained
    **XGBoost machine learning model** and generate
    actionable customer retention recommendations.
    """
)

# =====================================================
# 2. LOAD TRAINED MODEL
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR / "models" / "customer_churn_xgboost.pkl"
)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Save the trained model from your notebook first."
        )

    return joblib.load(MODEL_PATH)


try:
    artifact = load_model()
    model = artifact["model"]
    threshold = artifact["threshold"]

except Exception as error:
    st.error(f"Unable to load the model: {error}")
    st.stop()


# =====================================================
# 3. MODEL FEATURE CONFIGURATION
# =====================================================

model_features = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "TenureGroup",
    "TotalServices",
    "HasSupport",
    "HasSecurity"
]


# =====================================================
# 4. FEATURE ENGINEERING
# =====================================================

def engineer_features(data):

    data = data.copy()

    # Tenure Group
    data["TenureGroup"] = pd.cut(
        data["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=[
            "New Customer",
            "Short Term",
            "Medium Term",
            "Long Term"
        ]
    )

    # Total Number of Services
    service_columns = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    data["TotalServices"] = data[
        service_columns
    ].apply(
        lambda row: sum(value == "Yes" for value in row),
        axis=1
    )

    # Support availability
    data["HasSupport"] = (
        data["TechSupport"] == "Yes"
    ).astype(int)

    # Security availability
    data["HasSecurity"] = (
        data["OnlineSecurity"] == "Yes"
    ).astype(int)

    return data


# =====================================================
# 5. RETENTION RECOMMENDATIONS
# =====================================================

def generate_recommendations(customer):

    recommendations = []

    if customer["Contract"] == "Month-to-month":
        recommendations.append(
            "Offer a discounted one-year or two-year contract."
        )

    if customer["tenure"] <= 12:
        recommendations.append(
            "Provide an onboarding offer to improve early engagement."
        )

    if customer["TechSupport"] == "No":
        recommendations.append(
            "Offer a technical support package."
        )

    if customer["OnlineSecurity"] == "No":
        recommendations.append(
            "Promote an online security service."
        )

    if customer["PaymentMethod"] == "Electronic check":
        recommendations.append(
            "Encourage automatic payment methods for convenience."
        )

    if customer["MonthlyCharges"] >= 80:
        recommendations.append(
            "Review the customer's plan and offer a suitable pricing option."
        )

    if customer["TotalServices"] <= 2:
        recommendations.append(
            "Explore relevant additional services that may improve engagement."
        )

    if not recommendations:
        recommendations.append(
            "Maintain engagement through loyalty rewards and regular feedback."
        )

    return recommendations


# =====================================================
# 6. CUSTOMER INPUT FORM
# =====================================================

st.header("👤 Customer Information")

st.info(
    "Enter the customer's details below. "
    "Engineered features will be calculated automatically."
)

with st.form("customer_form"):

    col1, col2, col3 = st.columns(3)

    # -----------------------------------------------
    # PERSONAL DETAILS
    # -----------------------------------------------

    with col1:

        st.subheader("Personal Details")

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        senior = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        partner = st.selectbox(
            "Has Partner",
            ["No", "Yes"]
        )

        dependents = st.selectbox(
            "Has Dependents",
            ["No", "Yes"]
        )

        tenure = st.slider(
            "Tenure (Months)",
            min_value=0,
            max_value=72,
            value=12
        )

    # -----------------------------------------------
    # SERVICE DETAILS
    # -----------------------------------------------

    with col2:

        st.subheader("Service Details")

        phone = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "No phone service", "Yes"]
        )

        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["No", "No internet service", "Yes"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["No", "No internet service", "Yes"]
        )

        device_protection = st.selectbox(
            "Device Protection",
            ["No", "No internet service", "Yes"]
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["No", "No internet service", "Yes"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["No", "No internet service", "Yes"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["No", "No internet service", "Yes"]
        )

    # -----------------------------------------------
    # ACCOUNT DETAILS
    # -----------------------------------------------

    with col3:

        st.subheader("Account Details")

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        paperless = st.selectbox(
            "Paperless Billing",
            ["No", "Yes"]
        )

        payment = st.selectbox(
            "Payment Method",
            [
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Electronic check",
                "Mailed check"
            ]
        )

        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=200.0,
            value=70.0,
            step=1.0
        )

        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=10000.0,
            value=840.0,
            step=10.0
        )

    st.divider()

    submitted = st.form_submit_button(
        "🔍 Predict Customer Churn",
        use_container_width=True
    )


# =====================================================
# 7. PREDICTION
# =====================================================

if submitted:

    # Create customer input
    customer = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    input_data = pd.DataFrame([customer])

    # Generate engineered features
    input_data = engineer_features(input_data)

    # Match exact model feature order
    input_data = input_data[model_features]

    # Predict churn probability
    probabilities = model.predict_proba(input_data)[0]

    # Find the probability corresponding to "Yes"
    classes = list(model.classes_)

    if "Yes" in classes:
        yes_index = classes.index("Yes")
    else:
        yes_index = 1

    churn_probability = float(probabilities[yes_index])

    # Apply selected threshold
    prediction = (
        "Yes"
        if churn_probability >= threshold
        else "No"
    )

    # =================================================
    # 8. DISPLAY RESULTS
    # =================================================

    st.divider()
    st.header("📈 Prediction Results")

    result1, result2, result3 = st.columns(3)

    with result1:
        st.metric(
            "Churn Probability",
            f"{churn_probability * 100:.2f}%"
        )

    with result2:
        st.metric(
            "Classification Threshold",
            f"{threshold:.4f}"
        )

    with result3:
        st.metric(
            "Predicted Churn",
            prediction
        )

    st.progress(churn_probability)

    if prediction == "Yes":
        st.error(
            "⚠️ High Churn Risk: This customer is predicted to churn."
        )
    else:
        st.success(
            "✅ Lower Churn Risk: This customer is predicted to stay."
        )

    # =================================================
    # 9. CUSTOMER PROFILE
    # =================================================

    st.subheader("Customer Profile")

    profile1, profile2, profile3 = st.columns(3)

    with profile1:
        st.metric(
            "Tenure Group",
            str(input_data["TenureGroup"].iloc[0])
        )

    with profile2:
        st.metric(
            "Total Services",
            int(input_data["TotalServices"].iloc[0])
        )

    with profile3:
        st.metric(
            "Monthly Charges",
            f"${monthly_charges:.2f}"
        )

    # =================================================
    # 10. RETENTION RECOMMENDATIONS
    # =================================================

    st.divider()
    st.subheader("💡 Customer Retention Recommendations")

    recommendations = generate_recommendations(
        customer | {
            "TotalServices": int(
                input_data["TotalServices"].iloc[0]
            )
        }
    )

    for recommendation in recommendations:
        st.write("✔️", recommendation)

    # =================================================
    # 11. INPUT FEATURE DETAILS
    # =================================================

    with st.expander("View Model Input Features"):

        st.dataframe(
            input_data,
            use_container_width=True
        )

    st.caption(
        "Predictions are estimates based on the trained model. "
        "Retention recommendations are rule-based suggestions, "
        "not guaranteed outcomes."
    )


# =====================================================
# 12. FOOTER
# =====================================================

st.divider()

st.markdown(
    """
    **AI-Powered Customer Churn Prediction and Retention System**

    Built using Python, Pandas, Scikit-learn, XGBoost and Streamlit.
    """
)
