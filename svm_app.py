import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config("Smart Loan Approval System", layout="centered")

st.title("🏦 Smart Loan Approval System")

st.markdown(
    "This system uses **Support Vector Machines (SVM)** to predict loan approval."
)

df = pd.read_csv("loan_train.csv")
df = df.dropna()

df = df[[
    "ApplicantIncome",
    "LoanAmount",
    "Credit_History",
    "Self_Employed",
    "Property_Area",
    "Loan_Status"
]]

le = LabelEncoder()
df["Self_Employed"] = le.fit_transform(df["Self_Employed"])
df["Property_Area"] = le.fit_transform(df["Property_Area"])
df["Loan_Status"] = le.fit_transform(df["Loan_Status"])

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

st.sidebar.header("Applicant Details")

income = st.sidebar.number_input("Applicant Income", min_value=0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)

credit = st.sidebar.radio("Credit History", ["Yes", "No"])
credit_val = 1 if credit == "Yes" else 0

employment = st.sidebar.selectbox(
    "Employment Status",
    ["Self Employed", "Not Self Employed"]
)
emp_val = 1 if employment == "Self Employed" else 0

property_area = st.sidebar.selectbox(
    "Property Area",
    ["Urban", "Semiurban", "Rural"]
)

prop_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}
prop_val = prop_map[property_area]

kernel = st.radio(
    "Select SVM Kernel",
    ["Linear SVM", "Polynomial SVM", "RBF SVM"]
)

if kernel == "Linear SVM":
    model = SVC(kernel="linear", probability=True)
elif kernel == "Polynomial SVM":
    model = SVC(kernel="poly", degree=3, probability=True)
else:
    model = SVC(kernel="rbf", probability=True)

model.fit(X_train, y_train)

if st.button("Check Loan Eligibility"):

    user_data = np.array([[
        income,
        loan_amount,
        credit_val,
        emp_val,
        prop_val
    ]])

    user_data = scaler.transform(user_data)

    prediction = model.predict(user_data)[0]
    confidence = model.predict_proba(user_data).max() * 100

    if prediction == 1:
        st.success("✅ Loan Approved")
        st.info(
            "Based on credit history and income pattern, "
            "the applicant is likely to repay the loan."
        )
    else:
        st.error("❌ Loan Rejected")
        st.info(
            "Based on income and credit behavior, "
            "the applicant is unlikely to repay the loan."
        )

    st.write("Kernel Used:", kernel)
    st.write(f"Confidence Score: {confidence:.2f}%")

