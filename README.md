# 🏠 House Price Prediction System

An end-to-end Machine Learning system for predicting property prices based on tabular house attributes. Built using Python, Scikit-Learn, Flask REST API, and Streamlit.

🚀 **Live Interactive Web App**: [https://anchalkushwaha-12-house-pridector-frontendapp-iiqofr.streamlit.app/](https://anchalkushwaha-12-house-pridector-frontendapp-iiqofr.streamlit.app/)

---

## 📁 Project Structure

```
house_price_prediction/
├── data/
│   ├── generate_dataset.py       # Script to generate synthetic tabular dataset
│   └── house_prices.csv          # Generated CSV dataset (1,200 samples)
├── models/
│   ├── train_model.py            # Model training & evaluation pipeline
│   ├── house_price_model.joblib  # Serialized best ML model pipeline artifact
│   └── metrics.json              # Model evaluation metrics JSON summary
├── backend/
│   ├── app.py                    # Flask REST API server (/health, /predict)
│   └── test_api.py               # Automated unit tests for API endpoints
├── frontend/
│   └── app.py                    # Streamlit interactive Web UI dashboard
├── reports/
│   ├── generate_report.py        # Report generator script (standalone HTML)
│   ├── final_report.html         # Final analytical project report with embedded charts
│   └── assets/                   # Feature importance & diagnostic plots
├── requirements.txt              # Project Python dependencies
└── README.md                     # Project documentation
```

---

## 📊 Dataset Schema

The dataset contains 8 input features and 1 continuous target (`Price`):

| Feature Column | Type | Description / Range |
| :--- | :--- | :--- |
| `Area(sqft)` | Numeric | Property square footage (500 – 5,000 sqft) |
| `Rooms` | Numeric | Total number of bedrooms (1 – 6) |
| `Bathrooms` | Numeric | Total number of bathrooms (1 – 5) |
| `Floors` | Numeric | Number of floors (1 – 3) |
| `Location` | Categorical | Neighborhood (`Rural`, `Suburban`, `Urban`, `Downtown`) |
| `YearBuilt` | Numeric | Construction year (1975 – 2024) |
| `Parking` | Numeric | Vehicle parking spaces (0 – 3) |
| `Condition` | Categorical | Property physical condition (`Fair`, `Good`, `Excellent`) |
| **`Price`** | Target | Property valuation in USD ($) |

---

## 🤖 Model Performance Summary

Four regression algorithms were trained and evaluated on an 80/20 test split:

| Model Algorithm | $R^2$ Score | MAE ($) | RMSE ($) | MAPE (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Gradient Boosting Regressor (Selected)** | **0.9901** | **$23,447.48** | **$31,017.94** | **3.82%** |
| Random Forest Regressor | 0.9740 | $38,758.54 | $50,337.07 | 6.27% |
| Linear Regression | 0.9311 | $59,527.26 | $81,950.33 | 10.99% |
| Ridge Regression | 0.9300 | $59,910.05 | $82,569.27 | 10.94% |

---

## 🌐 Live Web Demo & Cloud Deployment
- **Live Streamlit Web App**: [https://anchalkushwaha-12-house-pridector-frontendapp-iiqofr.streamlit.app/](https://anchalkushwaha-12-house-pridector-frontendapp-iiqofr.streamlit.app/)

---

## 🚀 Quickstart & Local Execution Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset (Optional - pre-generated)
```bash
python data/generate_dataset.py
```

### 3. Train Machine Learning Models
```bash
python models/train_model.py
```

### 4. Start Flask REST API Backend
```bash
python backend/app.py
```
*API will run on `http://127.0.0.1:5000`.*

### 5. Launch Streamlit Web UI Frontend
In a separate terminal:
```bash
streamlit run frontend/app.py
```
*Streamlit dashboard will open at `http://localhost:8501`.*

### 6. Generate Analytical Project Report
```bash
python reports/generate_report.py
```
*Open `reports/final_report.html` in your browser.*

---

## 🔌 API Documentation

### Health Check
`GET /health`
```json
{
  "model_loaded": true,
  "status": "healthy"
}
```

### Predict Valuation
`POST /predict`

**Request Body:**
```json
{
  "Area(sqft)": 2400,
  "Rooms": 3,
  "Bathrooms": 2,
  "Floors": 2,
  "Location": "Suburban",
  "YearBuilt": 2015,
  "Parking": 2,
  "Condition": "Good"
}
```

**Response Payload:**
```json
{
  "status": "success",
  "predicted_price": 644305.77,
  "formatted_price": "$644,305.77",
  "price_per_sqft": 268.46,
  "confidence_interval": {
    "lower_bound": 619822.15,
    "upper_bound": 668789.39
  }
}
```
