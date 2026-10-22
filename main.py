"""
House Price Prediction API - FastAPI Backend

Run:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

Endpoints:
    GET  /health          - Health check
    GET  /metadata        - Feature ranges and categories for form validation
    POST /predict         - Predict house price from property features
"""

from pathlib import Path
import json
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from preprocessing import CombinedAttributesAdder


BASE = Path(__file__).resolve().parent
MODEL_FILE = BASE / "final_model.pkl"
PIPELINE_FILE = BASE / "preprocessing_pipeline.pkl"
METADATA_FILE = BASE / "model_metadata.json"

# Global model storage
model = None
pipeline = None
metadata = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and pipeline on startup."""
    global model, pipeline, metadata

    if not (MODEL_FILE.exists() and PIPELINE_FILE.exists()):
        print("WARNING: Model files not found. Run train_model.py first.")
    else:
        model = joblib.load(MODEL_FILE)
        pipeline = joblib.load(PIPELINE_FILE)
        print("Model and pipeline loaded successfully.")

    if METADATA_FILE.exists():
        metadata = json.loads(METADATA_FILE.read_text())
        print("Metadata loaded successfully.")
    else:
        print("WARNING: Metadata file not found. Run train_model.py to generate it.")

    yield

    # Cleanup on shutdown
    model = None
    pipeline = None
    metadata = {}


app = FastAPI(
    title="House Price Prediction API",
    description="ML-powered property valuation using California housing dataset",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class PropertyFeatures(BaseModel):
    """Input features for house price prediction."""
    longitude: float = Field(..., description="Longitude coordinate")
    latitude: float = Field(..., description="Latitude coordinate")
    housing_median_age: float = Field(..., description="Median age of houses in the block")
    total_rooms: float = Field(..., description="Total number of rooms")
    total_bedrooms: float = Field(..., description="Total number of bedrooms")
    population: float = Field(..., description="Population of the block")
    households: float = Field(..., description="Number of households")
    median_income: float = Field(..., description="Median income (in tens of thousands)")
    ocean_proximity: str = Field(..., description="Proximity to ocean")


class PredictionResponse(BaseModel):
    """Prediction result with confidence interval."""
    predicted_price: float = Field(..., description="Predicted house price in USD")
    lower_bound: float = Field(..., description="Lower bound of 95% prediction interval")
    upper_bound: float = Field(..., description="Upper bound of 95% prediction interval")
    confidence: float = Field(..., description="Model confidence percentage (0-100)")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    metadata_loaded: bool


class MetadataResponse(BaseModel):
    """Metadata for form validation."""
    numeric_ranges: dict
    ocean_proximity_categories: list[str]
    target: str
    test_rmse: float | None = None
    train_rows: int | None = None
    test_rows: int | None = None


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        metadata_loaded=bool(metadata),
    )


@app.get("/metadata", response_model=MetadataResponse)
async def get_metadata():
    """Get feature ranges and categories for form validation."""
    if not metadata:
        raise HTTPException(status_code=503, detail="Metadata not available. Run train_model.py first.")

    return MetadataResponse(
        numeric_ranges=metadata.get("numeric_ranges", {}),
        ocean_proximity_categories=metadata.get("ocean_proximity_categories", []),
        target=metadata.get("target", "median_house_value"),
        test_rmse=metadata.get("test_rmse"),
        train_rows=metadata.get("train_rows"),
        test_rows=metadata.get("test_rows"),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: PropertyFeatures):
    """Predict house price from property features."""
    if model is None or pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check server logs.")

    try:
        # Create input DataFrame
        input_df = pd.DataFrame([{
            "longitude": features.longitude,
            "latitude": features.latitude,
            "housing_median_age": features.housing_median_age,
            "total_rooms": features.total_rooms,
            "total_bedrooms": features.total_bedrooms,
            "population": features.population,
            "households": features.households,
            "median_income": features.median_income,
            "ocean_proximity": features.ocean_proximity,
        }])

        # Transform using preprocessing pipeline
        prepared = pipeline.transform(input_df)

        # Main prediction
        prediction = float(model.predict(prepared)[0])

        # Confidence estimation from tree agreement
        if hasattr(model, "estimators_") and len(model.estimators_) > 0:
            tree_predictions = np.array([
                float(tree.predict(prepared)[0])
                for tree in model.estimators_
            ])

            prediction = float(np.mean(tree_predictions))
            prediction_std = float(np.std(tree_predictions))

            # Approximate 95% prediction interval
            lower_bound = max(0.0, prediction - 1.96 * prediction_std)
            upper_bound = prediction + 1.96 * prediction_std

            # Relative uncertainty -> confidence
            relative_uncertainty = prediction_std / max(abs(prediction), 1.0)
            confidence = 100.0 * (1.0 - min(relative_uncertainty, 0.95))
            confidence = max(0.0, min(confidence, 100.0))
        else:
            lower_bound = prediction
            upper_bound = prediction
            confidence = 0.0

        return PredictionResponse(
            predicted_price=prediction,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence=confidence,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)