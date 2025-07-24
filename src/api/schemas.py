from pydantic import BaseModel, Field
from typing import List, Literal

class HousePredictionRequest(BaseModel):
    sqft: float = Field(..., gt=1000, lt=5000, description="Square footage of the house")
    bedrooms: int = Field(..., ge=1, le=6, description="Number of bedrooms")
    bathrooms: float = Field(..., gt=0.5, le=5.0, description="Number of bathrooms")
    location: Literal["Rural", "Suburb", "Urban", "Downtown", "Waterfront", "Mountain"] = Field(..., description="Location type")
    year_built: int = Field(..., ge=1945, le=2023, description="Year the house was built")
    condition: Literal["Poor", "Fair", "Good", "Excellent"] = Field(..., description="Condition of the house")
    price_per_sqft: float = Field(..., gt=50, lt=1000, description="Expected price per square foot in your area (e.g., 320)")

    class Config:
        schema_extra = {
            "example": {
                "sqft": 1527,
                "bedrooms": 2,
                "bathrooms": 1.5,
                "location": "Suburb",
                "year_built": 1956,
                "condition": "Good",
                "price_per_sqft": 320
            }
        }

class PredictionResponse(BaseModel):
    predicted_price: float = Field(..., description="Predicted house price in dollars")
    confidence_interval: List[float] = Field(..., description="90% confidence interval [lower, upper]")
    features_importance: dict = Field(default={}, description="Feature importance scores")
    prediction_time: str = Field(..., description="Timestamp of the prediction")

    class Config:
        schema_extra = {
            "example": {
                "predicted_price": 489650.75,
                "confidence_interval": [440685.68, 538615.82],
                "features_importance": {},
                "prediction_time": "2025-07-24T12:30:45.123456"
            }
        }