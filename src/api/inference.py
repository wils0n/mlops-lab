import joblib
import pandas as pd
import os
from datetime import datetime
from pathlib import Path
from .schemas import HousePredictionRequest, PredictionResponse

# Get the project root directory (three levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "trained" / "house_price_model.pkl"
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "trained" / "preprocessor.pkl"

try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print(f"✅ Model loaded: {type(model).__name__}")
    print(f"✅ Preprocessor loaded successfully")
except Exception as e:
    raise RuntimeError(f"Error loading model or preprocessor: {str(e)}")

def predict_price(request: HousePredictionRequest) -> PredictionResponse:
    """
    Predict house price based on input features.
    """
    try:
        # ✅ PASO 1: Crear DataFrame con datos base
        input_data = pd.DataFrame([{
            'sqft': request.sqft,
            'bedrooms': request.bedrooms,
            'bathrooms': request.bathrooms,
            'location': request.location,
            'year_built': request.year_built,
            'condition': request.condition
        }])
        
        # ✅ PASO 2: Calcular features derivadas
        current_year = datetime.now().year
        input_data['house_age'] = current_year - input_data['year_built']
        input_data['bed_bath_ratio'] = input_data['bedrooms'] / input_data['bathrooms']
        
        # ✅ PASO 3: Agregar price_per_sqft como estimación inicial
        location_price_estimates = {
            'Rural': 180,
            'Suburb': 320,
            'Urban': 280,
            'Downtown': 350,
            'Waterfront': 450,
            'Mountain': 250
        }
        
        condition_multipliers = {
            'Poor': 0.7,
            'Fair': 0.85,
            'Good': 1.0,
            'Excellent': 1.3
        }
        
        # Calcular price_per_sqft estimado
        base_price_per_sqft = location_price_estimates.get(request.location, 300)
        condition_multiplier = condition_multipliers.get(request.condition, 1.0)
        estimated_price_per_sqft = base_price_per_sqft * condition_multiplier
        
        input_data['price_per_sqft'] = estimated_price_per_sqft
        
        # ✅ PASO 4: Debug - Mostrar las columnas
        print(f"Input columns: {input_data.columns.tolist()}")
        print(f"Input data:\n{input_data}")
        
        # ✅ PASO 5: Aplicar preprocessing
        processed_features = preprocessor.transform(input_data)
        
        # ✅ PASO 6: Hacer predicción
        predicted_price = model.predict(processed_features)[0]
        
        # Convert to Python float and round
        predicted_price = round(float(predicted_price), 2)
        
        # Confidence interval (10% range)
        confidence_interval = [
            round(predicted_price * 0.9, 2), 
            round(predicted_price * 1.1, 2)
        ]
        
        return PredictionResponse(
            predicted_price=predicted_price,
            confidence_interval=confidence_interval,
            features_importance={},
            prediction_time=datetime.now().isoformat()
        )
        
    except Exception as e:
        print(f"Error details: {str(e)}")
        raise Exception(f"Prediction failed: {str(e)}")

def batch_predict(requests: list[HousePredictionRequest]) -> list[PredictionResponse]:
    """
    Perform batch predictions.
    """
    return [predict_price(req) for req in requests]