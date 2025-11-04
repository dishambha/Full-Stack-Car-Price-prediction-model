import sys
import os
import threading
import time
import requests

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import numpy as np
import pandas as pd
import joblib
import json
import warnings
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Literal, Dict, Any, List
from sqlalchemy.orm import Session
from database import init_db, get_db, PredictionLog

# Suppress warnings
warnings.filterwarnings("ignore")

# --- 1. INITIAL SETUP ---
MODELS_DIR = os.path.join(current_dir, "models")
MODELS_METADATA_PATH = os.path.dirname(current_dir)

models = {}
metadata = {}
owner_to_score = {'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3, 'Fourth & Above Owner': 4, 'Test Drive Car': 0}

app = FastAPI(
    title="IVPP - Indian Vehicle Price Predictor API",
    version="1.0.0",
    description="Backend service for predicting used car and bike prices."
)

# --- KEEP SERVER ALIVE (Render fix) ---
RENDER_APP_URL = os.getenv("RENDER_EXTERNAL_URL", None)  # Set this as environment variable on Render

def keep_server_alive():
    """Background thread that pings the server every 10 minutes to prevent Render sleep."""
    if not RENDER_APP_URL:
        print("⚠️ RENDER_APP_URL not set — keep-alive disabled.")
        return
    while True:
        try:
            requests.get(f"{RENDER_APP_URL}/api/v1/health", timeout=5)
            print("💡 Keep-alive ping sent.")
        except Exception as e:
            print(f"⚠️ Keep-alive ping failed: {e}")
        time.sleep(600)  # every 10 minutes


@app.on_event("startup")
def load_artifacts_and_init_db():
    global models, metadata
    try:
        models['car'] = joblib.load(f'{MODELS_DIR}/rf_best_car.pkl')
        models['bike'] = joblib.load(f'{MODELS_DIR}/rf_best_bike.pkl')
        with open(f'{MODELS_METADATA_PATH}/ui_metadata.json', 'r') as f:
            metadata = json.load(f)
        init_db()
        print("✅ Artifacts and database initialized successfully.")

        # Start keep-alive thread
        threading.Thread(target=keep_server_alive, daemon=True).start()

    except Exception as e:
        print(f"❌ ERROR: Failed to load artifacts or initialize DB. Error: {e}")
        raise RuntimeError("Critical startup process failed.") from e


# --- 2. UTILITY FUNCTIONS ---
def get_km_category(km_driven: int) -> str:
    if km_driven < 20000: return 'Low_Mileage'
    elif km_driven < 60000: return 'Medium_Mileage'
    elif km_driven < 120000: return 'High_Mileage'
    else: return 'Very_High_Mileage'

def transform_input(data: 'PredictionRequest', vehicle_type: Literal['car', 'bike']) -> pd.DataFrame:
    feature_cols = metadata[f'{vehicle_type}_feature_cols']
    df_input = pd.DataFrame(0, index=[0], columns=feature_cols)
    df_input['age'] = data.age
    df_input['owner_score'] = owner_to_score.get(data.owner_type, 1.0)
    features_to_encode = {
        'brand': data.brand, 'fuel': data.fuel, 'seller_type': data.seller_type,
        'transmission': data.transmission, 'km_category': get_km_category(data.km_driven)
    }
    for feature_type, feature_value in features_to_encode.items():
        col_name = f'{feature_type}_{feature_value}'
        if col_name in df_input.columns:
            df_input[col_name] = 1
    return df_input

def predict_with_range(model, X_input: pd.DataFrame, confidence_level=2) -> Dict[str, float]:
    predictions = [np.expm1(tree.predict(X_input)) for tree in model.estimators_]
    predictions = np.array(predictions).flatten()
    mean_price = predictions.mean()
    std_dev = predictions.std()
    lower_bound = max(0, mean_price - confidence_level * std_dev)
    upper_bound = mean_price + confidence_level * std_dev
    return {'mean_price': round(mean_price, 0), 'min_price': round(lower_bound, 0), 'max_price': round(upper_bound, 0)}

def get_feature_importance(model, feature_list: list) -> List[Dict[str, float]]:
    importance_df = pd.DataFrame({
        'feature': feature_list,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    def clean_name(name):
        for prefix in ['brand_', 'fuel_', 'seller_type_', 'transmission_', 'km_category_']:
            if name.startswith(prefix):
                name = name[len(prefix):]
        return name.replace('_', ' ').title()
    top_features = importance_df.head(10).copy()
    top_features['feature'] = top_features['feature'].apply(clean_name)
    top_features['importance'] = (top_features['importance'] * 100).round(2)
    return top_features.to_dict(orient='records')

def generate_depreciation_curve(model, base_input: pd.DataFrame, current_age: int) -> List[Dict[str, float]]:
    ages = range(1, 21)
    curve_data = []
    input_df = base_input.copy()
    for age in ages:
        input_df['age'] = age
        mean_p = np.expm1(model.predict(input_df))[0]
        curve_data.append({'age': age, 'price': round(mean_p, 0), 'is_current': (age == current_age)})
    return curve_data

def log_prediction_to_db(db: Session, request: 'PredictionRequest', results: Dict[str, float]):
    log_entry = PredictionLog(
        vehicle_type=request.vehicle_type,
        input_features=request.model_dump_json(),
        predicted_mean=results['mean_price'],
        predicted_min=results['min_price'],
        predicted_max=results['max_price']
    )
    db.add(log_entry)
    db.commit()
    db.refresh(log_entry)


# --- 3. SCHEMAS ---
class PredictionRequest(BaseModel):
    vehicle_type: Literal['car', 'bike']
    brand: str
    age: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner_type: str

class PredictionResponse(BaseModel):
    mean_price: float
    min_price: float
    max_price: float
    confidence_level: float = 0.95
    feature_importance: List[Dict[str, Any]]
    depreciation_curve: List[Dict[str, Any]]


# --- 4. ENDPOINTS ---
@app.get("/api/v1/health")
def health_check():
    if not models or not metadata:
        raise HTTPException(status_code=503, detail="Service Unavailable.")
    return {"status": "ok", "message": "IVPP API is running."}

@app.get("/api/v1/metadata")
def get_metadata():
    if not metadata:
        raise HTTPException(status_code=503, detail="Metadata not loaded.")
    return {
        'car_brands': metadata.get('car_brands', []),
        'bike_brands': metadata.get('bike_brands', []),
        'fuel': metadata.get('fuel', []),
        'transmission': metadata.get('transmission', []),
        'seller_type': metadata.get('seller_type', []),
        'owner_type': metadata.get('owner_type', []),
        'min_age': metadata.get('min_age', 1),
        'max_age': metadata.get('max_age', 20),
        'max_km': metadata.get('max_km', 300000)
    }

@app.post("/api/v1/predict", response_model=PredictionResponse)
def predict_price(request: PredictionRequest, db: Session = Depends(get_db)):
    vehicle_type = request.vehicle_type
    model = models.get(vehicle_type)
    try:
        X_input = transform_input(request, vehicle_type)
        price_results = predict_with_range(model, X_input, confidence_level=2)
        feature_list = metadata[f'{vehicle_type}_feature_cols']
        importance_list = get_feature_importance(model, feature_list)
        depreciation_data = generate_depreciation_curve(model, X_input, request.age)
        try:
            log_prediction_to_db(db, request, price_results)
        except Exception as e:
            print(f"⚠️ Warning: Failed to log prediction: {e}")
    except Exception as e:
        print(f"❌ CRITICAL PREDICTION ERROR: {e}")
        return PredictionResponse(
            mean_price=0.0, min_price=0.0, max_price=0.0,
            feature_importance=[], depreciation_curve=[]
        )
    return PredictionResponse(
        mean_price=price_results['mean_price'],
        min_price=price_results['min_price'],
        max_price=price_results['max_price'],
        feature_importance=importance_list,
        depreciation_curve=depreciation_data
    )
