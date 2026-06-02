# 📊 Customer Churn Prediction

A machine learning project to predict customer churn using the Telco Customer Churn dataset. This project includes data analysis, model training, and a Streamlit web application for making predictions.

## 📁 Project Structure

```
churn_customer/
├── project_deployment.py          # Streamlit web application
├── customer_churn_prediction.ipynb # Main notebook with model training
├── clustering-churn-and-correlation-analysis.ipynb  # Data analysis & clustering
├── churn_customer.sav             # Trained model (pickle format)
├── WA_Fn-UseC_-Telco-Customer-Churn.csv # Dataset
├── requirements.txt               # Python dependencies
└── README.md                       # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yehiayaser/churn_customer.git
   cd churn_customer
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Streamlit App

```bash
streamlit run project_deployment.py
```

The app will open in your browser at `http://localhost:8501`

## 📚 Files Description

### `project_deployment.py`
Streamlit web application for interactive customer churn predictions. Features:
- Model loading with error handling
- User-friendly interface
- Real-time predictions

### `customer_churn_prediction.ipynb`
Jupyter notebook containing:
- Data exploration and preprocessing
- Feature engineering
- Model training and evaluation
- Performance metrics and visualizations

### `clustering-churn-and-correlation-analysis.ipynb`
Advanced analysis including:
- Customer clustering analysis
- Correlation analysis between features
- Statistical insights

### `churn_customer.sav`
Pre-trained machine learning model saved using pickle. Compatible with scikit-learn 1.5.2+

### `WA_Fn-UseC_-Telco-Customer-Churn.csv`
The Telco Customer Churn dataset containing:
- 7,043 customer records
- 21 features including demographics and service usage
- Target variable: Churn (Yes/No)

## 📊 Dataset Info

**Source:** Telco Customer Churn Dataset

**Features:**
- Customer demographics (age, gender, etc.)
- Account information (tenure, contract, etc.)
- Services used (internet, phone, streaming, etc.)
- Monthly charges, total charges

**Target:** Customer churn (Yes/No)

## 🛠️ Dependencies

- **streamlit**: Web app framework
- **pandas**: Data manipulation
- **scikit-learn**: Machine learning library
- **numpy**: Numerical computations
- **matplotlib/seaborn**: Data visualization
- **jupyter**: Interactive notebooks

## 🔧 Troubleshooting

### PyArrow DLL Error
If you encounter a PyArrow DLL error on Windows:
```bash
pip install --only-binary :all: pyarrow==24.0.0
```

### Model Loading Error
Ensure you're running the app from the repository root directory where `churn_customer.sav` is located.

## 📈 Model Performance

The trained model achieves:
- High accuracy in predicting customer churn
- Balanced precision and recall
- See notebooks for detailed metrics and visualizations

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements!

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

**Yehia Yaser**  
GitHub: [@yehiayaser](https://github.com/yehiayaser)

---

**Last Updated:** June 2, 2026