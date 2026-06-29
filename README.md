# Credit Card Fraud Detection

🔗 **Live demo:** [fraud-detection-system-dckqfvr92hchnkhdsfn9mx.streamlit.app](https://fraud-detection-system-dckqfvr92hchnkhdsfn9mx.streamlit.app/)

A machine learning project that detects fraudulent credit card transactions using Logistic Regression, with a focus on correctly handling extreme class imbalance — and an interactive Streamlit app to demo predictions live.

## Problem Statement

Credit card fraud is rare: in this dataset, only **0.34% of transactions** are fraudulent (88 out of ~25,837 transactions). This extreme imbalance means a naive model that simply predicts "not fraud" every time would score **99.66% accuracy** while catching zero actual fraud — making accuracy a misleading metric for this problem.

This project focuses on building a model that correctly identifies the rare fraud cases, and evaluating it with metrics that actually reflect real-world performance: **precision and recall**, not just accuracy.

## Dataset

- Source: [Credit Card Fraud Detection (Kaggle, ULB Machine Learning Group)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Transactions made by European cardholders, anonymized via PCA transformation (features `V1`–`V28`), plus `Time` and `Amount`
- Target variable: `Class` (0 = legitimate, 1 = fraud)

## Approach

1. **Exploratory Data Analysis (EDA)** — examined class distribution and transaction amounts by class, uncovering the severe imbalance and a notable mean-vs-median gap in fraud transaction amounts (a few large fraud transactions skew the average, while most fraud transactions are actually small).
2. **Preprocessing** — handled a missing/corrupted row, scaled features using `StandardScaler` (necessary for Logistic Regression to converge properly).
3. **Train/test split** — used stratified splitting to preserve the same fraud ratio in both sets, since random splitting risked leaving the test set with too few (or zero) fraud examples.
4. **Modeling** — trained a Logistic Regression model with `class_weight='balanced'` to counteract the imbalance, rather than letting the model default to ignoring the rare class.
5. **Evaluation** — used a confusion matrix, precision, and recall instead of accuracy, since accuracy is misleading on imbalanced data.

## Results

| Metric (Fraud class) | Score |
|---|---|
| Recall | 100% (caught all fraud cases in the test set) |
| Precision | 21% |
| False Negatives | 0 |
| False Positives | 66 |

The model prioritizes **recall over precision** — it catches every fraud case in the test set, at the cost of some false alarms on legitimate transactions. This is a deliberate, justifiable tradeoff: in fraud detection, a missed fraud (false negative) typically costs real money directly, while a false alarm (false positive) usually just costs a customer a quick verification step. Missing fraud is generally far more expensive than over-flagging.

## Tech Stack

- **Python** — pandas, scikit-learn, joblib
- **Streamlit** — interactive web app for live predictions

## Project Structure

```
fraud-detection-project/
├── README.md
├── app.py                      # Streamlit app
├── fraud_model.pkl              # Trained Logistic Regression model
├── scaler.pkl                   # Fitted StandardScaler
├── sample_test_data.csv         # Sample test features for the demo app
├── sample_test_labels.csv       # Corresponding true labels
└── requirements.txt
```

## How to Run Locally

```bash
# Clone the repo
git clone https://github.com/Swanandik/fraud-detection-system.git
cd fraud-detection-system

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`. Pick a row number from the test set and click **Predict** to see the model's fraud prediction and confidence score, alongside the actual label.

## What I Learned

- Why accuracy is a misleading metric on imbalanced datasets, and how to reason about precision/recall tradeoffs in a business context
- How to handle class imbalance using `class_weight='balanced'` and stratified train/test splitting
- Why feature scaling matters for models like Logistic Regression
- Building and debugging an end-to-end ML pipeline, from raw data to a deployed interactive app
- Real-world debugging: environment setup, library version mismatches, and data cleaning issues

## Future Improvements

- Compare against other models (Random Forest, XGBoost)
- Add SMOTE-based oversampling as an alternative to class weighting
- Add data visualizations (class distribution, amount comparison) directly into the app
