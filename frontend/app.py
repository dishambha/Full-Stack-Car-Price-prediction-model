# frontend/app.py

import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List

# --- Configuration ---
FASTAPI_BASE_URL = "https://full-stack-car-price-prediction-model-1.onrender.com/api/v1"


# --- Utility Functions ---

@st.cache_data(ttl=3600)
def fetch_metadata():
    """Fetches static data (brands, fuel types, etc.) from the FastAPI metadata endpoint."""
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/metadata")
        response.raise_for_status() 
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error: Could not connect to backend API. Please ensure the FastAPI server is running at {FASTAPI_BASE_URL.split('/api')[0]}")
        st.stop()
        
def send_prediction_request(data: Dict[str, Any]) -> Dict[str, Any]:
    """Sends the user's input to the FastAPI prediction endpoint."""
    try:
        response = requests.post(f"{FASTAPI_BASE_URL}/predict", json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        st.error(f"Prediction failed with HTTP error: {response.status_code}. Details: {response.text}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error sending request to backend: {e}")
        return None

# --- Main Streamlit Application ---

def main():
    st.set_page_config(
        page_title="IVPP - Indian Vehicle Price Predictor",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("💰 IVPP: Indian Vehicle Price Predictor (3-Feature Model)")
    st.markdown("---")

    metadata = fetch_metadata()
    
    # --- Sidebar for Vehicle Type ---
    st.sidebar.header("Vehicle Selection")
    
    vehicle_type = st.sidebar.radio(
        "Select Vehicle Type",
        ("car", "bike")
    )
    
    # --- STABLE FIX: Use the simple brand lists for stable selection ---
    if vehicle_type == 'car':
        # Use the simple list of primary brands
        brand_list = metadata['car_brands']
    else:
        # Use the simple list of primary bike brands
        brand_list = metadata['bike_brands']
    
    # --- Input Form ---
    st.header(f"Input Features for {vehicle_type.upper()}")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        # Row 1
        with col1:
            # FIX: Reverting to the simple "Brand" label and list
            brand = st.selectbox("Brand", brand_list, key='brand')
        
        with col2:
            age = st.slider(
                "Age (Years)",
                min_value=metadata['min_age'],
                max_value=metadata['max_age'],
                value=5,
                step=1,
                key='age'
            )
            
        with col3:
            km_driven = st.number_input(
                "Kilometers Driven",
                min_value=0,
                max_value=metadata['max_km'],
                value=50000,
                step=1000,
                key='km_driven'
            )

        # Row 2 (Other Features)
        col4, col5, col6 = st.columns(3)
        
        with col4:
            fuel = st.selectbox("Fuel Type", metadata['fuel'], key='fuel')
            
        with col5:
            transmission = st.selectbox("Transmission", metadata['transmission'], key='transmission')
            
        with col6:
            owner_type = st.selectbox("Owner Type", metadata['owner_type'], key='owner_type')
        
        col7, col8 = st.columns(2)

        with col7:
            seller_type = st.selectbox("Seller Type", metadata['seller_type'], key='seller_type')
        
        with col8:
            st.empty()
        
        st.markdown("---")
        
        submitted = st.form_submit_button("Predict Price & Tell the Story")

    # --- Prediction Logic ---
    if submitted:
        payload = {
            "vehicle_type": vehicle_type, "brand": brand, "age": age,
            "km_driven": km_driven, "fuel": fuel, "seller_type": seller_type,
            "transmission": transmission, "owner_type": owner_type
        }
        
        with st.spinner(f"Predicting price for your {brand} {vehicle_type}..."):
            prediction_data = send_prediction_request(payload)

        if prediction_data:
            display_results(prediction_data, brand, age)

def display_results(data: Dict[str, Any], brand: str, current_age: int):
    """Displays the three storytelling sections."""
    
    st.markdown("## 🎯 Predicted Price & Confidence Range")
    
    col_mean, col_range = st.columns(2)
    
    with col_mean:
        st.metric(
            label="Predicted Mean Price (INR)",
            value=f"₹{data['mean_price']:,.0f}"
        )
        
    with col_range:
        st.info(
            f"**95% Confidence Range:**\n"
            f"Min Price: **₹{data['min_price']:,.0f}**\n"
            f"Max Price: **₹{data['max_price']:,.0f}**\n\n"
            f"**Story:** We are highly confident the fair market value lies within this range."
        )

    st.markdown("---")
    
    # --- Feature Importance (Storytelling Feature 2) ---
    st.markdown("## 🧠 Price Drivers: What Influenced the Price?")
    
    importance_df = pd.DataFrame(data['feature_importance'])
    st.bar_chart(importance_df.set_index('feature'))
    
    st.success(f"**Story:** The chart shows the biggest factors. Key indicators for the price are '{importance_df.iloc[0]['feature']}' and '{importance_df.iloc[1]['feature']}'.")
    
    st.markdown("---")

    # --- Depreciation Curve (Storytelling Feature 3) ---
    st.markdown("## 📉 Future Value: Depreciation Over Time")

    curve_df = pd.DataFrame(data['depreciation_curve'])
    curve_df.set_index('age', inplace=True)
    
    st.line_chart(curve_df['price'])
    
    future_age = current_age + 5
    if future_age in curve_df.index:
        future_price = curve_df.loc[future_age, 'price']
        st.warning(f"**Story:** Your vehicle's value is expected to be approximately **₹{future_price:,.0f}** in 5 years (Age {future_age}).")
    else:
         st.warning("Story: Value forecast is based on the line chart above.")
    
if __name__ == "__main__":
    main()