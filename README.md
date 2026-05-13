# 🛒 Advanced Sales Forecasting System

A machine learning-powered sales forecasting system built using real-world retail data.  
This project predicts future product sales using historical trends, promotions, transactions, seasonality, and engineered time-series features.

The system includes:
- 📊 Data preprocessing pipeline
- 🤖 XGBoost forecasting model
- 📈 Feature engineering for temporal patterns
- 🖥️ Interactive Streamlit dashboard
- 💾 Model serialization and reusable pipeline

---

# 🚀 Project Overview

Sales forecasting is a critical problem in retail and business analytics.  
Accurate forecasts help businesses:

- manage inventory efficiently
- reduce overstocking and losses
- improve staffing and operations
- optimize supply chain planning
- understand customer demand patterns

This project simulates a real-world forecasting workflow using a large-scale retail dataset containing millions of records.

---

# 📂 Dataset

Dataset used:
- **Store Sales Time Series Forecasting (Kaggle)**

The dataset includes:
- historical product sales
- store metadata
- oil prices
- transactions
- holidays/events
- promotions

---

# 🧠 Machine Learning Pipeline

## 1️⃣ Data Loading
Multiple datasets were loaded and combined:

- train.csv
- stores.csv
- oil.csv
- transactions.csv
- holidays_events.csv

---

## 2️⃣ Data Merging
The datasets were merged using:
- `store_nbr`
- `date`

This created a unified retail forecasting dataset.

---

## 3️⃣ Feature Engineering

### 📅 Time-Based Features
Extracted:
- year
- month
- day
- day of week
- week of year
- day of year
- weekend indicators

---

### ⏳ Lag Features
Historical dependencies were captured using:
- lag_1
- lag_7
- lag_14

These features help the model learn temporal patterns in sales.

---

### 📈 Rolling Statistics
Rolling window features were created to smooth short-term fluctuations:

- rolling_mean_7
- rolling_mean_14
- rolling_median_7
- rolling_std_14

---

### 🛍️ Promotion Features
Promotion-aware features included:

- promo_last_7
- promo_x_transactions

These capture the impact of promotions on demand.

---

### 📊 Behavioral Features
Additional business-oriented features:

- momentum_7
- dow_avg
- store_avg
- transaction-based indicators

---

# 🤖 Model Used

## XGBoost Regressor

The final forecasting model was built using:

- XGBoost
- Regularization
- Time-aware train/test split
- One-hot encoding
- Feature alignment

---

# 📉 Evaluation Metrics

The model was evaluated using:

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

### Final Results

| Metric | Value |
|--------|------|
| RMSE | ~238 |
| MAE | ~61 |

These results were achieved on a large-scale real-world dataset containing millions of rows.

---

# 🖥️ Streamlit Dashboard

The project includes an interactive forecasting dashboard where users can:

- select store and product category
- choose prediction date
- input promotion and transaction data
- generate future sales forecasts
- visualize historical sales trends

---

# 📊 Key Insights

The model learned that:
- recent sales history strongly influences future demand
- promotions significantly impact sales
- temporal behavior matters more than static information
- rolling averages are highly predictive in retail forecasting

---

# 🏗️ Project Structure

```bash
FUTURE_ML_01/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── outputs/
│   ├── xgboost_model.pkl
│   └── feature_columns.pkl
│
├── src/
│   └── app.py
│
├── requirements.txt
│
└── README.md
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone <your-repo-link>
cd FUTURE_ML_01
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run src/app.py
```

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- Streamlit

---

# 📌 Future Improvements

Potential future upgrades:

- LSTM / Deep Learning forecasting
- Multi-step forecasting
- Real-time API deployment
- Automated retraining pipeline
- Cloud deployment
- Demand anomaly detection

---

# 🎯 Learning Outcomes

This project helped demonstrate:
- large-scale data preprocessing
- real-world feature engineering
- time-series forecasting concepts
- model optimization
- deployment-oriented ML workflows
- end-to-end ML system design

---

# 👩‍💻 Author

**Charishma Ganta**  
Machine Learning & Cybersecurity Student  
Focused on AI systems, IoT, and real-world ML applications.
