import pickle
import streamlit as st 
import pandas as pd
import os

# Load the model
model_path = r'E:\Yehia ( Study )\AI and Data Science\Projects\Machine Learning projects\Churn customer\churn_customer.sav'

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

    st.title("Customer Churn Prediction")
