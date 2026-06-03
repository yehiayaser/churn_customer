import pickle
import streamlit as st 
import pandas as pd
import numpy as np
from pathlib import Path

# Get the directory of the current script
BASE_DIR = Path(__file__).parent

# Load the model
model_path = BASE_DIR / 'churn_customer_coefficients.sav'
data_path = BASE_DIR / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    model = None

# Load and preprocess dataset to get expected feature names
try:
    df_data = pd.read_csv(data_path)
    
    # Apply same preprocessing as training
    df_data.drop('customerID', axis=1, inplace=True)
    df_data['TotalCharges'] = pd.to_numeric(df_data['TotalCharges'], errors='coerce').fillna(0)
    df_data.drop_duplicates(inplace=True)
    
    # Binary encoding
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'SeniorCitizen']
    for col in binary_cols:
        if col in df_data.columns:
            if col == 'gender':
                df_data[col] = (df_data[col] == 'Male').astype(int)
            else:
                df_data[col] = (df_data[col] == 'Yes').astype(int)
    
    # One-hot encoding
    ohe_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
    df_data = pd.get_dummies(df_data, columns=ohe_cols, drop_first=False).astype(int)
    
    # Get Churn column and remove it
    if 'Churn' in df_data.columns:
        df_data.drop('Churn', axis=1, inplace=True)
    
    # Feature engineering - Enhanced
    df_data['ChargesPerMonth'] = df_data['TotalCharges'] / (df_data['tenure'] + 1)
    df_data['Charge_x_Tenure'] = df_data['MonthlyCharges'] * df_data['tenure']
    df_data['TenureBucket'] = np.where(df_data['tenure'] <= 12, 0, 
                                       np.where(df_data['tenure'] <= 24, 1, 
                                       np.where(df_data['tenure'] <= 48, 2, 3)))
    
    service_cols = [col for col in df_data.columns if col.endswith('_Yes')]
    df_data['TotalServices'] = df_data[service_cols].sum(axis=1)
    
    # Additional advanced features
    df_data['MonthlyToTotalRatio'] = df_data['MonthlyCharges'] / (df_data['TotalCharges'] + 1)
    df_data['ChargeAcceleration'] = (df_data['MonthlyCharges'] - df_data['ChargesPerMonth']) / (df_data['ChargesPerMonth'] + 1)
    df_data['AvgMonthlySpend'] = df_data['TotalCharges'] / (df_data['tenure'] + 1)
    df_data['ServiceAdoptionRate'] = df_data['TotalServices'] / 13  # 13 possible services
    df_data['TenureMonths'] = df_data['tenure']
    df_data['ChargeCategory'] = pd.cut(df_data['MonthlyCharges'], bins=[0, 30, 60, 100, 200], labels=[0, 1, 2, 3])
    df_data['TenureCategory'] = pd.cut(df_data['tenure'], bins=[-1, 6, 12, 24, 72], labels=[0, 1, 2, 3])
    df_data['IsHighValue'] = (df_data['TotalCharges'] > df_data['TotalCharges'].median()).astype(int)
    df_data['IsNewCustomer'] = (df_data['tenure'] <= 3).astype(int)
    df_data['MonthlyCost_Per_Service'] = df_data['MonthlyCharges'] / (df_data['TotalServices'] + 1)
    
    # Get expected feature names and order
    EXPECTED_FEATURES = df_data.columns.tolist()
    
except Exception as e:
    st.error(f"❌ Error loading dataset: {e}")
    EXPECTED_FEATURES = None

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
    Ensures all expected features exist in the correct order
    """
    # Create DataFrame
    df = pd.DataFrame([input_dict])
    
    # Convert Yes/No to 1/0 for binary features
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'SeniorCitizen']
    for col in binary_cols:
        if col in df.columns:
            if col == 'gender':
                df[col] = (df[col] == 'Male').astype(int)
            else:
                df[col] = (df[col] == 'Yes').astype(int)
    
    # One-hot encode categorical features
    ohe_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=False).astype(int)
    
    # Create engineered features - Enhanced
    tenure = input_dict['tenure']
    monthly = input_dict['MonthlyCharges']
    total = input_dict['TotalCharges']
    
    df['ChargesPerMonth'] = total / (tenure + 1) if tenure > 0 else 0
    df['Charge_x_Tenure'] = monthly * tenure
    df['TenureBucket'] = np.where(tenure <= 12, 0, np.where(tenure <= 24, 1, np.where(tenure <= 48, 2, 3)))
    
    # TotalServices: count of active services
    service_cols = [col for col in df.columns if col.endswith('_Yes')]
    df['TotalServices'] = df[service_cols].sum(axis=1)
    
    # Advanced features
    df['MonthlyToTotalRatio'] = monthly / (total + 1)
    df['ChargeAcceleration'] = (monthly - (total / (tenure + 1))) / ((total / (tenure + 1)) + 1)
    df['AvgMonthlySpend'] = total / (tenure + 1)
    df['ServiceAdoptionRate'] = df['TotalServices'] / 13
    df['TenureMonths'] = tenure
    df['ChargeCategory'] = np.where(monthly <= 30, 0, np.where(monthly <= 60, 1, np.where(monthly <= 100, 2, 3)))
    df['TenureCategory'] = np.where(tenure <= 6, 0, np.where(tenure <= 12, 1, np.where(tenure <= 24, 2, 3)))
    df['IsHighValue'] = 1 if total > 3000 else 0  # Threshold based on median
    df['IsNewCustomer'] = 1 if tenure <= 3 else 0
    df['MonthlyCost_Per_Service'] = monthly / (df['TotalServices'] + 1)
    
    # ✅ CRITICAL FIX: Ensure all expected features exist with correct order
    if EXPECTED_FEATURES is not None:
        # Add missing columns with 0 values
        for col in EXPECTED_FEATURES:
            if col not in df.columns:
                df[col] = 0
        
        # Reorder columns to match training data exactly
        df = df[EXPECTED_FEATURES]
    
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

# Display engineered features
st.subheader("⚙️ Engineered Features")
st.write("""
The model automatically computes these advanced features from your input:
- **ChargesPerMonth**: Average monthly charge (TotalCharges / Tenure)
- **Charge_x_Tenure**: Interaction between monthly charges and tenure
- **MonthlyToTotalRatio**: Proportion of total charges that are monthly
- **ChargeAcceleration**: Rate of charge increase over time
- **ServiceAdoptionRate**: Percentage of available services customer uses
- **IsNewCustomer**: Whether customer is within first 3 months
- **IsHighValue**: Whether customer's total charges are above median
- **ChargeCategory**: Monthly charge bracket (Low/Medium/High/Premium)
- **TenureCategory**: Customer loyalty category based on months with company
- **MonthlyCost_Per_Service**: Average cost per active service
""")

# Predict button
if st.sidebar.button("🔮 Predict Churn", key="predict_button"):
    if model is not None:
        try:
            # Preprocess the input
            df_processed = preprocess_input(input_values)
            
            # Make prediction
            prediction = model.predict(df_processed)[0]
            prediction_proba = model.predict_proba(df_processed)[0]
            
            # Extract feature values for insights
            tenure = input_values['tenure']
            monthly_charges = input_values['MonthlyCharges']
            total_charges = input_values['TotalCharges']
            
            # Display result with enhanced insights
            st.divider()
            st.header("🔮 Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error("⚠️ **HIGH CHURN RISK**")
                    churn_risk = prediction_proba[1] * 100
                    retention_rate = 100 - churn_risk
                    st.metric("Churn Risk Score", f"{churn_risk:.1f}%")
                    st.metric("Retention Rate", f"{retention_rate:.1f}%")
                else:
                    st.success("✅ **LOW CHURN RISK**")
                    churn_risk = prediction_proba[1] * 100
                    retention_rate = 100 - churn_risk
                    st.metric("Churn Risk Score", f"{churn_risk:.1f}%")
                    st.metric("Retention Rate", f"{retention_rate:.1f}%")
            
            with col2:
                st.write("**Customer Profile Summary:**")
                st.write(f"• Tenure: {tenure} months")
                st.write(f"• Monthly Charges: ${monthly_charges:.2f}")
                st.write(f"• Total Charges: ${total_charges:.2f}")
                service_count = sum([1 for v in input_values.values() if v == 'Yes'])
                st.write(f"• Active Services: {service_count}")
            
            # Insights based on engineered features
            st.subheader("💡 Key Insights")
            
            if tenure <= 3:
                st.info("⏰ **New Customer Alert**: Customer is within first 3 months. Early engagement crucial!")
            
            if tenure > 24:
                st.success("👑 **Loyal Customer**: Long-term relationship. Priority retention candidate!")
            
            if monthly_charges > 100:
                st.warning(f"💰 **High Value Customer**: Monthly spend (${monthly_charges:.2f}) above average. Churn would be costly!")
            
            if total_charges > 5000:
                st.info(f"💼 **Premium Customer**: High lifetime value (${total_charges:.2f}). Deserves VIP support.")
            
            service_adoption = service_count / 13
            if service_adoption < 0.3:
                st.warning("📱 **Low Service Adoption**: Customer using <30% of available services. Cross-sell opportunity!")
            elif service_adoption > 0.7:
                st.success("🌟 **High Engagement**: Customer using >70% of services. Strong account health!")
                
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
            st.info(f"Debug: {str(e)}")
    else:
        st.error("❌ Model not loaded")

