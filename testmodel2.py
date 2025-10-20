import streamlit as st
import pandas as pd
import joblib


model = joblib.load('churn_model.pkl')
accuracy = joblib.load('model_accuracy.pkl')
encoders = joblib.load('labelencoders.pkl')   # order = ['Gender','Subscription Type','Contract Length','Churn']


st.title("Customer Churn Prediction App")
st.write("Predict whether a customer will churn or stay using your trained model")

# User inputs
age = st.number_input("Age", min_value=10, max_value=100, value=30)
gender = st.selectbox("Gender", encoders[0].classes_)  # 👈 pull options from encoder
tenure = st.number_input("Tenure (Months)", min_value=0, value=12)
usage_freq = st.number_input("Usage Frequency", min_value=0.0, value=10.0)
support_calls = st.number_input("Support Calls", min_value=0, value=2)
payment_delay = st.number_input("Payment Delay (Days)", min_value=0.0, value=5.0)
subscription_type = st.selectbox("Subscription Type", encoders[1].classes_)
contract_length = st.selectbox("Contract Length", encoders[2].classes_)
total_spend = st.number_input("Total Spend", min_value=0.0, value=500.0)
last_interaction = st.number_input("Last Interaction (Days Ago)", min_value=0.0, value=10.0)


input_data = pd.DataFrame({
    'Age': [age],
    'Gender': [encoders[0].transform([gender])[0]],
    'Tenure': [tenure],
    'Usage Frequency': [usage_freq],
    'Support Calls': [support_calls],
    'Payment Delay': [payment_delay],
    'Subscription Type': [encoders[1].transform([subscription_type])[0]],
    'Contract Length': [encoders[2].transform([contract_length])[0]],
    'Total Spend': [total_spend],
    'Last Interaction': [last_interaction]
})


if st.button("🔍 Predict Churn"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error(" The customer is likely to CHURN!")
    else:
        st.success(" The customer will likely STAY!")

    st.info(f" Model Accuracy (from training): {accuracy:.2f}%")
