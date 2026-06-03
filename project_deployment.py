import pickle
import streamlit as st 
import pandas as pd
import os
from pathlib import Path

# Get the directory of the current script
BASE_DIR = Path(__file__).parent

# Load the model using relative path
model_path = BASE_DIR / 'churn_customer.sav'
data_path = BASE_DIR / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    model = None

# Load dataset to analyze features
try:
    df_data = pd.read_csv(data_path)
except Exception as e:
    st.error(f"❌ Error loading dataset: {e}")
    df_data = None

st.title("📊 Customer Churn Prediction")
st.info('Easy Application For Customer Churn Prediction')

st.sidebar.header("🔧 Feature Selection")

# Define features with hints
features_info = {
    'gender': {
        'type': 'categorical',
        'options': ['Male', 'Female'],
        'help': 'Categorical: Choose Male or Female'
    },
    'SeniorCitizen': {
        'type': 'categorical',
        'options': ['Yes', 'No'],
        'help': 'Categorical: Is the customer a senior citizen? (Yes/No)'
    },
    'Partner': {
        'type': 'categorical',
        'options': ['Yes', 'No'],
        'help': 'Categorical: Does the customer have a partner? (Yes/No)'
    },
    'Dependents': {
        'type': 'categorical',
        'options': ['Yes', 'No'],
        'help': 'Categorical: Does the customer have dependents? (Yes/No)'
    },
    'Tenure': {
        'type': 'numeric',
        'min': 0,
        'max': 72,
        'help': 'Numeric: Number of months the customer has been with the company (0-72 months)'
    },
    'PhoneService': {
        'type': 'categorical',
        'options': ['Yes', 'No'],
        'help': 'Categorical: Does the customer have phone service? (Yes/No)'
    },
    'MultipleLines': {
        'type': 'categorical',
        'options': ['Yes', 'No', 'No phone service'],
        'help': 'Categorical: Does the customer have multiple phone lines? (Yes/No/No phone service)'
    },
    'InternetService': {
        'type': 'categorical',
        'options': ['DSL', 'Fiber optic', 'No'],
        'help': 'Categorical: Type of internet service (DSL/Fiber optic/No)'
    },
    'OnlineSecurity': {
        'type': 'categorical',
        'options': ['Yes', 'No', 'No internet service'],
        'help': 'Categorical: Does the customer have online security? (Yes/No/No internet service)'
    },
    'OnlineBackup': {
        'type': 'categorical',
        'options': ['Yes', 'No', 'No internet service'],
        'help': 'Categorical: Does the customer have online backup? (Yes/No/No internet service)'
    },
    'DeviceProtection': {
        'type': 'categorical',
        'options': ['Yes', 'No', 'No internet service'],
        'help': 'Categorical: Does the customer have device protection? (Yes/No/No internet service)'
    },
    'TechSupport': {
        'type': 'categorical',
        'options': ['Yes', 'No', 'No internet service'],
        'help': 'Categorical: Does the customer have tech support? (Yes/No/No internet service)'
    },
    'StreamingTV': {
        'type': 'categorical',
        'options': ['Yes', 'No', 'No internet service'],
        'help': 'Categorical: Does the customer have streaming TV service? (Yes/No/No internet service)'
    },
    'StreamingMovies': {
        'type': 'categorical',
        'options': ['Yes', 'No', 'No internet service'],
        'help': 'Categorical: Does the customer have streaming movies service? (Yes/No/No internet service)'
    },
    'Contract': {
        'type': 'categorical',
        'options': ['Month-to-month', 'One year', 'Two year'],
        'help': 'Categorical: Contract term (Month-to-month/One year/Two year)'
    },
    'PaperlessBilling': {
        'type': 'categorical',
        'options': ['Yes', 'No'],
        'help': 'Categorical: Does the customer have paperless billing? (Yes/No)'
    },
    'PaymentMethod': {
        'type': 'categorical',
        'options': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
        'help': 'Categorical: Payment method used by the customer'
    },
    'MonthlyCharges': {
        'type': 'numeric',
        'min': 0,
        'max': 150,
        'help': 'Numeric: Monthly charges in dollars (0-150)'
    }
}

# Create input fields with hints
input_values = {}

for feature, info in features_info.items():
    if info['type'] == 'categorical':
        input_values[feature] = st.sidebar.selectbox(
            label=f"📋 {feature}",
            options=info['options'],
            help=info['help']
        )
    else:  # numeric
        input_values[feature] = st.sidebar.number_input(
            label=f"🔢 {feature}",
            min_value=info['min'],
            max_value=info['max'],
            value=info['min'],
            help=info['help']
        )

# Create DataFrame from inputs
df = pd.DataFrame({
    'gender': [input_values['gender']],
    'SeniorCitizen': [input_values['SeniorCitizen']],
    'Partner': [input_values['Partner']],
    'Dependents': [input_values['Dependents']],
    'Tenure': [input_values['Tenure']],
    'PhoneService': [input_values['PhoneService']],
    'MultipleLines': [input_values['MultipleLines']],
    'InternetService': [input_values['InternetService']],
    'OnlineSecurity': [input_values['OnlineSecurity']],
    'OnlineBackup': [input_values['OnlineBackup']],
    'DeviceProtection': [input_values['DeviceProtection']],
    'TechSupport': [input_values['TechSupport']],
    'StreamingTV': [input_values['StreamingTV']],
    'StreamingMovies': [input_values['StreamingMovies']],
    'Contract': [input_values['Contract']],
    'PaperlessBilling': [input_values['PaperlessBilling']],
    'PaymentMethod': [input_values['PaymentMethod']],
    'MonthlyCharges': [input_values['MonthlyCharges']]
}, index=[0])

# Display feature info in main area
st.subheader("📌 Feature Information")
col1, col2 = st.columns(2)

with col1:
    st.write("**Categorical Features:**")
    categorical_features = [f for f, info in features_info.items() if info['type'] == 'categorical']
    for feat in categorical_features:
        st.write(f"• {feat}: {', '.join(features_info[feat]['options'])}")

with col2:
    st.write("**Numeric Features:**")
    numeric_features = [f for f, info in features_info.items() if info['type'] == 'numeric']
    for feat in numeric_features:
        min_val = features_info[feat]['min']
        max_val = features_info[feat]['max']
        st.write(f"• {feat}: {min_val} - {max_val}")

# Predict button
if st.sidebar.button("🔮 Predict", key="predict_button"):
    if model is not None:
        try:
            result = model.predict(df)[0]
            if result == 1:
                st.sidebar.success("⚠️ Customer will CHURN")
            else:
                st.sidebar.success("✅ Customer will NOT CHURN")
        except Exception as e:
            st.sidebar.error(f"❌ Prediction error: {e}")
    else:
        st.sidebar.error("❌ Model not loaded")

