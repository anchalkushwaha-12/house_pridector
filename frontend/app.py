import os
import requests
import joblib
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background-color: #F3F4F6;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #2563EB;
        margin-bottom: 1rem;
    }
    .price-display {
        font-size: 2.2rem;
        font-weight: 800;
        color: #059669;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    </style>
""", unsafe_allow_html=True)

# Helper Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "house_price_model.joblib")
ASSETS_DIR = os.path.join(BASE_DIR, "reports", "assets")
FLASK_API_URL = "http://127.0.0.1:5000/predict"

@st.cache_resource
def load_local_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading local model: {str(e)}")
            return None
    return None

local_pipeline = load_local_model()

# Header Section
st.markdown('<div class="main-header">🏠 AI House Price Prediction Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Estimate property market values instantly using machine learning regression</div>', unsafe_allow_html=True)

# Sidebar Inputs
st.sidebar.header("📋 Property Specifications")
st.sidebar.markdown("Adjust parameters to estimate property valuation.")

area_sqft = st.sidebar.slider("Total Area (sqft)", min_value=500, max_value=5000, value=2200, step=50)

col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    rooms = st.number_input("Bedrooms", min_value=1, max_value=8, value=3)
    floors = st.selectbox("Floors", options=[1, 2, 3], index=1)
with col_sb2:
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=6, value=2)
    parking = st.selectbox("Parking Spaces", options=[0, 1, 2, 3], index=1)

location = st.sidebar.selectbox("Location / Neighborhood", options=["Rural", "Suburban", "Urban", "Downtown"], index=1)
year_built = st.sidebar.slider("Year Built", min_value=1975, max_value=2024, value=2015)
condition = st.sidebar.radio("Property Condition", options=["Fair", "Good", "Excellent"], index=1, horizontal=True)

# Payload Construction
input_payload = {
    "Area(sqft)": area_sqft,
    "Rooms": rooms,
    "Bathrooms": bathrooms,
    "Floors": floors,
    "Location": location,
    "YearBuilt": year_built,
    "Parking": parking,
    "Condition": condition
}

# Prediction Execution
predict_clicked = st.sidebar.button("✨ Estimate Valuation", type="primary", use_container_width=True)

tab_pred, tab_analytics, tab_about = st.tabs(["📊 Price Estimate", "📈 Model Analytics", "ℹ️ Project Info"])

with tab_pred:
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.subheader("Valuation Summary")
        
        # Attempt Flask API call with local pipeline fallback
        predicted_price = None
        price_per_sqft = None
        lower_bound = None
        upper_bound = None
        source_mode = "Flask REST API"

        try:
            response = requests.post(FLASK_API_URL, json=input_payload, timeout=2)
            if response.status_code == 200:
                res_data = response.json()
                predicted_price = res_data["predicted_price"]
                price_per_sqft = res_data["price_per_sqft"]
                lower_bound = res_data["confidence_interval"]["lower_bound"]
                upper_bound = res_data["confidence_interval"]["upper_bound"]
            else:
                raise Exception(f"API Error {response.status_code}")
        except Exception:
            # Fallback to local model pipeline
            source_mode = "Direct ML Model Engine (Local Fallback)"
            if local_pipeline is not None:
                input_df = pd.DataFrame([input_payload])
                predicted_price = float(local_pipeline.predict(input_df)[0])
                predicted_price = round(max(0, predicted_price), 2)
                price_per_sqft = round(predicted_price / area_sqft, 2)
                margin = predicted_price * 0.038
                lower_bound = round(max(0, predicted_price - margin), 2)
                upper_bound = round(predicted_price + margin, 2)

        if predicted_price is not None:
            st.markdown(f"""
                <div class="metric-box">
                    <div class="stat-label">Estimated Market Price</div>
                    <div class="price-display">${predicted_price:,.2f}</div>
                    <div style="margin-top: 0.5rem; font-size: 0.95rem; color: #4B5563;">
                        Expected Range: <strong>${lower_bound:,.2f}</strong> – <strong>${upper_bound:,.2f}</strong> (±3.8%)
                    </div>
                </div>
            """, unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Price per Sq Ft", f"${price_per_sqft:.2f}")
            m2.metric("Age of Property", f"{2026 - year_built} years")
            m3.metric("Valuation Source", source_mode.split()[0])

            st.caption(f"Inference performed via: `{source_mode}`")

        else:
            st.warning("Unable to generate prediction. Please ensure Flask server is running or model pipeline exists.")

    with col_right:
        st.subheader("Selected Specifications")
        spec_df = pd.DataFrame({
            "Feature": ["Area (sqft)", "Location", "Bedrooms", "Bathrooms", "Floors", "Year Built", "Parking Spaces", "Condition"],
            "Value": [f"{area_sqft:,} sqft", location, rooms, bathrooms, floors, year_built, parking, condition]
        })
        st.dataframe(spec_df, hide_index=True, use_container_width=True)

with tab_analytics:
    st.subheader("Model Diagnostic & Feature Insights")
    
    col_img1, col_img2 = st.columns(2)
    
    act_pred_img = os.path.join(ASSETS_DIR, "actual_vs_predicted.png")
    feat_imp_img = os.path.join(ASSETS_DIR, "feature_importance.png")
    
    with col_img1:
        if os.path.exists(act_pred_img):
            st.image(act_pred_img, caption="Actual vs Predicted Prices", use_container_width=True)
        else:
            st.info("Actual vs Predicted plot available after training script execution.")

    with col_img2:
        if os.path.exists(feat_imp_img):
            st.image(feat_imp_img, caption="Feature Importance Ranking", use_container_width=True)
        else:
            st.info("Feature importance plot available after training script execution.")

with tab_about:
    st.subheader("About the Project")
    st.markdown("""
    This House Price Prediction application leverages a Gradient Boosting ML pipeline trained on historical property data.
    
    - **Architecture**: Modular Python structure with Flask API backend & Streamlit frontend.
    - **Model Accuracy**: $R^2 = 0.9901$, MAE = \$23,447.48 on evaluation test set.
    - **Dataset Features**: `Area(sqft)`, `Rooms`, `Bathrooms`, `Floors`, `Location`, `YearBuilt`, `Parking`, `Condition`.
    """)
