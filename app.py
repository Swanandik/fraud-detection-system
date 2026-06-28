import streamlit as st
import joblib
import pandas as pd

st.title("Credit Card Fraud Detection")
st.write("This app predicts whether a transaction is fraudulent.")

model = joblib.load('fraud_model.pkl')
scaler = joblib.load('scaler.pkl')

X_test_sample = pd.read_csv('sample_test_data.csv')
y_test_sample = pd.read_csv('sample_test_labels.csv')

st.write("Pick a transaction to test:")
index = st.number_input("Enter a row number (0 to {})".format(len(X_test_sample)-1), min_value=0, max_value=len(X_test_sample)-1, value=0)

selected_row = X_test_sample.iloc[[index]]
actual_label = y_test_sample.iloc[index, 0]

st.write("Selected transaction features:")
st.write(selected_row)

if st.button("Predict"):
    selected_row_scaled = scaler.transform(selected_row)
    prediction = model.predict(selected_row_scaled)[0]
    probability = model.predict_proba(selected_row_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Predicted: FRAUD (confidence: {probability:.2%})")
    else:
        st.success(f"✅ Predicted: Not Fraud (confidence: {1-probability:.2%})")

    st.write(f"Actual label was: {'Fraud' if actual_label == 1 else 'Not Fraud'}")