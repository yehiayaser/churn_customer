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

st.title("📊 Customer Churn Prediction")
