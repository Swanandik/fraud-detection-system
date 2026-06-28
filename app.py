import streamlit as st
import joblib
import pandas as pd

# Must be the very first Streamlit command
st.set_page_config(
    page_title="Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# --- Custom CSS for a richer look ---
st.markdown("""
<style>
.stApp {
    background-color: #0F172A;
}

h1, h2, h3, p, span, label, .stMarkdown {
    color: #E2E8F0 !important;
}

.block-container {
    padding-top: 2rem;
}

/* Card-style containers */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1E293B 0%, #243B55 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.2rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}

div[data-testid="stMetricLabel"] {
    color: #94A3B8 !important;
}

div[data-testid="stMetricValue"] {
    color: #4FD1C5 !important;
}

/* Subheaders */
h2, h3 {
    border-left: 4px solid #4FD1C5;
    padding-left: 0.6rem;
}

/* Buttons */
.stButton button {
    background: linear-gradient(135deg, #4FD1C5 0%, #2E86AB 100%);
    color: #0F172A;
    font-weight: 600;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
}

.stButton button:hover {
    opacity: 0.9;
    color: #0F172A;
}

/* Expander */
div[data-testid="stExpander"] {
    background-color: #1E293B;
    border-radius: 10px;
    border: 1px solid #334155;
}

/* Dataframe */
div[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}

hr {
    border-color: #334155;
}
</style>
""", unsafe_allow_html=True)

# Load model and data
model = joblib.load('fraud_model.pkl')
scaler = joblib.load('scaler.pkl')
X_test_sample = pd.read_csv('sample_test_data.csv')
y_test_sample = pd.read_csv('sample_test_labels.csv')

# --- Header ---
st.title("💳 Credit Card Fraud Detection")
st.caption("Logistic Regression model trained to catch rare fraud cases in imbalanced transaction data.")

st.divider()

# --- Top metric cards ---
total = len(y_test_sample)
fraud_count = int(y_test_sample.iloc[:, 0].sum())
fraud_rate = fraud_count / total * 100

col1, col2, col3 = st.columns(3)
col1.metric("Total Transactions (test set)", f"{total:,}")
col2.metric("Fraud Cases", fraud_count)
col3.metric("Fraud Rate", f"{fraud_rate:.2f}%")

st.divider()

# --- Two-column layout: input on left, chart on right ---
left, right = st.columns([1, 1])

with left:
    st.subheader("🔎 Test a transaction")
    st.write("Pick a row from the test set and see what the model predicts.")

    index = st.number_input(
        f"Row number (0 to {len(X_test_sample) - 1})",
        min_value=0,
        max_value=len(X_test_sample) - 1,
        value=0
    )

    selected_row = X_test_sample.iloc[[index]]
    actual_label = y_test_sample.iloc[index, 0]

    with st.expander("View transaction features"):
        st.dataframe(selected_row, use_container_width=True)

    if st.button("🔍 Predict", type="primary", use_container_width=True):
        selected_row_scaled = scaler.transform(selected_row)
        prediction = model.predict(selected_row_scaled)[0]
        probability = model.predict_proba(selected_row_scaled)[0][1]

        if prediction == 1:
            st.error(f"⚠️ **Predicted: FRAUD** (confidence: {probability:.2%})")
        else:
            st.success(f"✅ **Predicted: Not Fraud** (confidence: {1 - probability:.2%})")

        actual_text = "Fraud" if actual_label == 1 else "Not Fraud"
        st.info(f"Actual label: **{actual_text}**")

with right:
    st.subheader("📊 Class distribution")
    st.write("Fraud is extremely rare — this is the core challenge of the project.")

    chart_data = pd.DataFrame({
        "Class": ["Not Fraud", "Fraud"],
        "Count": [total - fraud_count, fraud_count]
    })
    st.bar_chart(chart_data.set_index("Class"), color="#4FD1C5")

    st.caption(
        "A model that always predicts 'Not Fraud' would still score "
        f"**{(total - fraud_count) / total * 100:.2f}% accuracy** — which is why "
        "this project evaluates with precision/recall instead of accuracy."
    )

st.divider()

# --- Model performance section ---
st.subheader("📈 Model performance on test set")
perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
perf_col1.metric("Recall (Fraud)", "100%")
perf_col2.metric("Precision (Fraud)", "21%")
perf_col3.metric("Frauds Missed", "0")
perf_col4.metric("False Alarms", "66")

st.caption(
    "The model is tuned to prioritize catching every fraud case (recall), "
    "even at the cost of some false alarms — a deliberate tradeoff, since "
    "missing real fraud is typically far more costly than a false alarm."
)