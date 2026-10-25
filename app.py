import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Loan Eligibility Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .approved-card {
        background-color: #ECFDF5;
        border: 2px solid #10B981;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-top: 15px;
    }
    .rejected-card {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-top: 15px;
    }
    /* Disable any truncation or ellipsis (...) */
    * {
        text-overflow: clip !important;
    }
    div[data-testid="stMetric"], 
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricLabel"] *,
    div[data-testid="stMetricValue"] * {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        word-break: normal !important;
    }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "loan_pipeline.joblib")

# ---------------------------------------------------------
# Model Loader
# ---------------------------------------------------------
@st.cache_resource
def load_pipeline():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

pipeline = load_pipeline()

# ---------------------------------------------------------
# Sidebar - Project Information & Presets
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank-building.png", width=70)
    st.title("Project Overview")
    st.markdown("""
    **Pure Machine Learning Application**
    - **Dataset**: Kaggle / Analytics Vidhya
    - **Pipeline**: Scikit-Learn `ColumnTransformer`
    - **Techniques**:
      - Median imputation + StandardScaler
      - Mode imputation + OneHotEncoder
    """)
    st.divider()

    st.subheader("⚡ Quick Presets")
    preset = st.radio(
        "Auto-populate fields:",
        options=["Custom Input", "🎯 Ideal Candidate (Low Risk)", "⚠️ High Risk Candidate"],
        index=0
    )
    st.caption("Select a preset above to quickly test approval vs rejection.")

# Default values based on preset
if preset == "🎯 Ideal Candidate (Low Risk)":
    default_gender = "Male"
    default_married = "Yes"
    default_dependents = "1"
    default_education = "Graduate"
    default_self_emp = "No"
    default_app_income = 7500
    default_coapp_income = 3500
    default_loan_amt = 130
    default_loan_term = 360
    default_credit_hist = "Meets guidelines (1.0)"
    default_property = "Semiurban"
elif preset == "⚠️ High Risk Candidate":
    default_gender = "Male"
    default_married = "No"
    default_dependents = "3+"
    default_education = "Not Graduate"
    default_self_emp = "Yes"
    default_app_income = 1800
    default_coapp_income = 0
    default_loan_amt = 320
    default_loan_term = 360
    default_credit_hist = "Does NOT meet guidelines (0.0)"
    default_property = "Rural"
else:
    default_gender = "Female"
    default_married = "Yes"
    default_dependents = "0"
    default_education = "Graduate"
    default_self_emp = "No"
    default_app_income = 5400
    default_coapp_income = 1800
    default_loan_amt = 150
    default_loan_term = 360
    default_credit_hist = "Meets guidelines (1.0)"
    default_property = "Urban"

# ---------------------------------------------------------
# Main Content
# ---------------------------------------------------------
st.markdown('<div class="main-header">🏦 Loan Eligibility Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated credit evaluation powered by a Scikit-Learn Machine Learning pipeline.</div>', unsafe_allow_html=True)

col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("📋 Applicant Details")
    
    with st.expander("👤 Personal & Demographics", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            gender = st.selectbox("Gender", ["Male", "Female"], index=0 if default_gender == "Male" else 1)
            married = st.selectbox("Marital Status", ["Yes", "No"], index=0 if default_married == "Yes" else 1)
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"], index=["0", "1", "2", "3+"].index(default_dependents))
        with c2:
            education = st.selectbox("Education", ["Graduate", "Not Graduate"], index=0 if default_education == "Graduate" else 1)
            self_employed = st.selectbox("Self Employed", ["No", "Yes"], index=0 if default_self_emp == "No" else 1)
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"], index=["Urban", "Semiurban", "Rural"].index(default_property))

    with st.expander("💰 Financials & Loan Request", expanded=True):
        c3, c4 = st.columns(2)
        with c3:
            app_income = st.number_input("Applicant Monthly Income ($)", min_value=0, max_value=150000, value=default_app_income, step=250)
            coapp_income = st.number_input("Co-Applicant Monthly Income ($)", min_value=0, max_value=100000, value=default_coapp_income, step=250)
        with c4:
            loan_amount = st.number_input("Loan Amount Requested ($ in Thousands)", min_value=10, max_value=1000, value=default_loan_amt, step=5, help="Example: 150 = $150,000")
            loan_term = st.selectbox("Loan Term (Months)", [120, 180, 240, 300, 360, 480], index=[120, 180, 240, 300, 360, 480].index(default_loan_term))


    with st.expander("📜 Credit History", expanded=True):
        credit_history_str = st.radio(
            "Credit History Status",
            options=["Meets guidelines (1.0)", "Does NOT meet guidelines (0.0)"],
            index=0 if "1.0" in default_credit_hist else 1,
            help="Credit history records whether previous credit obligations were fulfilled."
        )
        credit_history_val = 1.0 if "1.0" in credit_history_str else 0.0

    eval_btn = st.button("🔍 Evaluate Loan Eligibility", type="primary", use_container_width=True)

with col_output:
    st.subheader("📊 Financial Assessment")

    total_income = app_income + coapp_income
    loan_full_val = loan_amount * 1000
    income_to_loan = (loan_full_val / (total_income * 12)) if total_income > 0 else 999.0

    st.markdown(f"""
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 20px;">
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 18px 10px; text-align: center;">
            <div style="color: #64748B; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6px;">
                Total Income
            </div>
            <div style="color: #0F172A; font-size: 1.6rem; font-weight: 700; white-space: nowrap;">
                ${total_income:,.0f}
            </div>
        </div>
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 18px 10px; text-align: center;">
            <div style="color: #64748B; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6px;">
                Loan Requested
            </div>
            <div style="color: #0F172A; font-size: 1.6rem; font-weight: 700; white-space: nowrap;">
                ${loan_full_val:,.0f}
            </div>
        </div>
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 18px 10px; text-align: center;">
            <div style="color: #64748B; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6px;">
                Loan Term
            </div>
            <div style="color: #0F172A; font-size: 1.6rem; font-weight: 700; white-space: nowrap;">
                {int(loan_term)} Months
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    if eval_btn:
        if pipeline is None:
            st.error("Model file `loan_pipeline.joblib` not found. Please run the notebook `loan_prediction.ipynb` first to generate and save the model.")
        else:
            # Prepare applicant DataFrame matching training schema
            input_df = pd.DataFrame([{
                'Gender': gender,
                'Married': married,
                'Dependents': dependents,
                'Education': education,
                'Self_Employed': self_employed,
                'ApplicantIncome': float(app_income),
                'CoapplicantIncome': float(coapp_income),
                'LoanAmount': float(loan_amount),
                'Loan_Amount_Term': float(loan_term),
                'Credit_History': credit_history_val,
                'Property_Area': property_area
            }])

            # Make prediction
            prediction = pipeline.predict(input_df)[0]
            probability = pipeline.predict_proba(input_df)[0][1]

            if prediction == 1:
                st.markdown(f"""
                <div class="approved-card">
                    <h2 style="color: #065F46; margin: 0;">✅ LOAN APPROVED</h2>
                    <p style="color: #047857; font-size: 1.1rem; margin-top: 8px;">
                        The applicant meets the eligibility criteria with an approval confidence of <b>{probability:.1%}</b>.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="rejected-card">
                    <h2 style="color: #991B1B; margin: 0;">❌ LOAN REJECTED</h2>
                    <p style="color: #B91C1C; font-size: 1.1rem; margin-top: 8px;">
                        The applicant does not meet current lending guidelines (Approval probability: <b>{probability:.1%}</b>).
                    </p>
                </div>
                """, unsafe_allow_html=True)

            st.write("")
            st.write("**Approval Probability Gauge:**")
            st.progress(min(max(float(probability), 0.0), 1.0))

            # Explainable insights
            st.markdown("### 💡 Decision Factors")
            factors = []
            if credit_history_val == 1.0:
                factors.append("✅ **Positive Credit History**: Applicant fulfills minimum credit bureau criteria.")
            else:
                factors.append("⚠️ **Adverse Credit History**: Absence of positive credit history is the strongest rejection predictor.")

            est_monthly_installment = (loan_full_val / loan_term) if loan_term > 0 else 0
            dti_ratio = (est_monthly_installment / total_income * 100) if total_income > 0 else 100

            if dti_ratio > 35:
                factors.append(f"⚠️ **High Debt Burden**: The estimated monthly payment of **\\${est_monthly_installment:,.0f}** would consume **{dti_ratio:.0f}%** of monthly income (\\${total_income:,.0f}), exceeding typical risk thresholds.")
            else:
                factors.append(f"✅ **Manageable Repayment**: The estimated monthly installment of **\\${est_monthly_installment:,.0f}** accounts for **{dti_ratio:.0f}%** of monthly income (\\${total_income:,.0f}).")

            if property_area == "Semiurban":
                factors.append("ℹ️ **Semiurban Property**: Historically correlates with higher approval rates in this dataset.")

            for factor in factors:
                st.markdown(f"- {factor}")
    else:
        st.info("👈 Enter applicant information on the left and click **'Evaluate Loan Eligibility'** to run the prediction.")
