import pickle
import streamlit as st 
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Get the directory of the current script
BASE_DIR = Path(__file__).parent

# Load the model
model_path = BASE_DIR / 'churn_customer.sav'
data_path = BASE_DIR / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    model = None

# Load dataset to fit scalers
try:
    df_data = pd.read_csv(data_path)
    # Fit the StandardScaler on the full training data for scaling consistency
    scaler = StandardScaler()
    # Note: In production, you should save the scaler with the model
except Exception as e:
    st.error(f"❌ Error loading dataset: {e}")
    df_data = None
    scaler = None

st.title("📊 Customer Churn Prediction")
st.info('Predict customer churn using Machine Learning')

st.sidebar.header("🔧 Customer Information")

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
    'tenure': {
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
    },
    'TotalCharges': {
        'type': 'numeric',
        'min': 0,
        'max': 10000,
        'help': 'Numeric: Total charges in dollars (0-10000)'
    }
}

# Create input fields
input_values = {}

for feature in ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure']:
    info = features_info[feature]
    if info['type'] == 'categorical':
        input_values[feature] = st.sidebar.selectbox(
            label=f"📋 {feature}", options=info['options'], help=info['help'])
    else:
        input_values[feature] = st.sidebar.number_input(
            label=f"🔢 {feature}", min_value=info['min'], max_value=info['max'], value=info['min'], help=info['help'])

for feature in ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup']:
    info = features_info[feature]
    input_values[feature] = st.sidebar.selectbox(
        label=f"📋 {feature}", options=info['options'], help=info['help'])

for feature in ['DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract']:
    info = features_info[feature]
    input_values[feature] = st.sidebar.selectbox(
        label=f"📋 {feature}", options=info['options'], help=info['help'])

for feature in ['PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']:
    info = features_info[feature]
    if info['type'] == 'categorical':
        input_values[feature] = st.sidebar.selectbox(
            label=f"📋 {feature}", options=info['options'], help=info['help'])
    else:
        input_values[feature] = st.sidebar.number_input(
            label=f"🔢 {feature}", min_value=info['min'], max_value=info['max'], value=info['min'], help=info['help'])

# Preprocessing Function
def preprocess_input(input_dict):
    """
    Preprocess input data exactly as the model was trained
    """
    # Create DataFrame
    df = pd.DataFrame([input_dict])
    
    # Convert Yes/No to 1/0 for binary features (LabelEncoder style)
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'SeniorCitizen']
    for col in binary_cols:
        if col in df.columns:
            df[col] = (df[col] == 'Yes').astype(int) if col != 'gender' else (df[col] == 'Male').astype(int)
    
    # One-hot encode categorical features
    ohe_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=False)
    
    # Create engineered features
    tenure = input_dict['tenure']
    monthly = input_dict['MonthlyCharges']
    total = input_dict['TotalCharges']
    
    df['ChargesPerMonth'] = total / (tenure + 1) if tenure > 0 else 0
    df['Charge_x_Tenure'] = monthly * tenure
    df['TenureBucket'] = np.where(tenure <= 12, 0, np.where(tenure <= 24, 1, np.where(tenure <= 48, 2, 3)))
    
    # TotalServices: count of active services
    service_cols = [col for col in df.columns if col.endswith('_Yes')]
    df['TotalServices'] = df[service_cols].sum(axis=1)
    
    return df

# Display feature info
st.subheader("📌 Feature Information")
col1, col2 = st.columns(2)

with col1:
    st.write("**Categorical Features:**")
    categorical_features = [f for f, info in features_info.items() if info['type'] == 'categorical']
    for feat in categorical_features[:10]:
        st.write(f"• {feat}: {', '.join(features_info[feat]['options'][:2])}")

with col2:
    st.write("**Numeric Features:**")
    numeric_features = [f for f, info in features_info.items() if info['type'] == 'numeric']
    for feat in numeric_features:
        min_val = features_info[feat]['min']
        max_val = features_info[feat]['max']
        st.write(f"• {feat}: {min_val} - {max_val}")

# Predict button
if st.sidebar.button("🔮 Predict Churn", key="predict_button"):
    if model is not None:
        try:
            # Preprocess the input
            df_processed = preprocess_input(input_values)
            
            # Make prediction
            prediction = model.predict(df_processed)[0]
            prediction_proba = model.predict_proba(df_processed)[0]
            
            # Display result
            if prediction == 1:
                st.sidebar.success("⚠️ **CHURN RISK DETECTED**")
                st.sidebar.metric("Churn Probability", f"{prediction_proba[1]:.1%}")
            else:
                st.sidebar.success("✅ **NO CHURN RISK**")
                st.sidebar.metric("Retention Probability", f"{prediction_proba[0]:.1%}")
                
        except Exception as e:
            st.sidebar.error(f"❌ Prediction error: {e}")
            st.sidebar.info(f"Debug: {str(e)}")
    else:
        st.sidebar.error("❌ Model not loaded")

