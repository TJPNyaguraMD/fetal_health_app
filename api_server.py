"""
Fetal Health Classification - FastAPI REST API
This API provides endpoints for fetal health prediction using the trained model.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import pickle
import json
import numpy as np
from typing import List, Dict
import os

# Initialize FastAPI app
app = FastAPI(
    title="Fetal Health Classification API",
    description="API for predicting fetal health status from CTG features",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load model artifacts
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model')

try:
    with open(os.path.join(MODEL_DIR, 'gradient_boosting_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    
    with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    
    with open(os.path.join(MODEL_DIR, 'feature_names.json'), 'r') as f:
        feature_names = json.load(f)
    
    with open(os.path.join(MODEL_DIR, 'model_metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    print("Model artifacts loaded successfully!")
    
except FileNotFoundError as e:
    print(f"Warning: Could not load model artifacts: {e}")
    model = None
    scaler = None
    feature_names = []
    metadata = {}


# Request/Response models
class CTGFeatures(BaseModel):
    """Input features from CTG recording."""
    baseline_value: float = Field(..., description="Baseline fetal heart rate")
    accelerations: float = Field(..., description="Number of accelerations per second")
    fetal_movement: float = Field(..., description="Number of fetal movements per second")
    uterine_contractions: float = Field(..., description="Number of uterine contractions per second")
    light_decelerations: float = Field(..., description="Number of light decelerations per second")
    severe_decelerations: float = Field(..., description="Number of severe decelerations per second")
    prolongued_decelerations: float = Field(..., description="Number of prolonged decelerations per second")
    abnormal_short_term_variability: float = Field(..., description="Percentage of abnormal short term variability")
    mean_value_of_short_term_variability: float = Field(..., description="Mean value of short term variability")
    percentage_of_time_with_abnormal_long_term_variability: float = Field(..., description="Percentage of time with abnormal long term variability")
    mean_value_of_long_term_variability: float = Field(..., description="Mean value of long term variability")
    histogram_width: float = Field(None, description="Width of histogram (optional)")
    histogram_min: float = Field(None, description="Minimum of histogram (optional)")
    histogram_max: float = Field(None, description="Maximum of histogram (optional)")
    histogram_number_of_peaks: float = Field(..., description="Number of peaks in histogram")
    histogram_number_of_zeroes: float = Field(..., description="Number of zeroes in histogram")
    histogram_mode: float = Field(None, description="Mode of histogram (optional)")
    histogram_mean: float = Field(..., description="Mean of histogram")
    histogram_median: float = Field(None, description="Median of histogram (optional)")
    histogram_variance: float = Field(None, description="Variance of histogram (optional)")
    histogram_tendency: float = Field(None, description="Tendency of histogram (optional)")
    
    @validator('*', pre=True)
    def convert_to_float(cls, v):
        if v is None:
            return None
        return float(v)


class PredictionResponse(BaseModel):
    """Response model for prediction."""
    prediction: int = Field(..., description="Predicted class (1=Normal, 2=Suspect, 3=Pathological)")
    prediction_label: str = Field(..., description="Human-readable prediction label")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")
    confidence: float = Field(..., description="Confidence score (0-1)")
    risk_level: str = Field(..., description="Risk assessment")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_accuracy: float = None
    feature_count: int = None


# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Fetal Health Classification API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch_predict",
            "features": "/features"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status."""
    return HealthResponse(
        status="healthy" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
        model_accuracy=metadata.get('test_accuracy'),
        feature_count=len(feature_names)
    )


@app.get("/features")
async def get_features():
    """Get list of required features."""
    return {
        "features": feature_names,
        "feature_count": len(feature_names),
        "dropped_features": [
            "histogram_width", "histogram_min", "histogram_max",
            "histogram_variance", "histogram_median", "histogram_tendency", "histogram_mode"
        ]
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: CTGFeatures):
    """
    Predict fetal health status from CTG features.
    
    Returns prediction with confidence score and probabilities.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to dataframe format
        input_dict = features.dict()
        
        # Remove None values and features that were dropped during training
        dropped_features = ['histogram_width', 'histogram_min', 'histogram_max', 
                          'histogram_variance', 'histogram_median', 'histogram_tendency', 'histogram_mode']
        input_dict = {k: v for k, v in input_dict.items() 
                     if v is not None and k not in dropped_features}
        
        # Create feature array in correct order
        feature_array = np.array([[input_dict.get(f, 0) for f in feature_names]])
        
        # Scale features
        feature_array_scaled = scaler.transform(feature_array)
        
        # Make prediction
        prediction = model.predict(feature_array_scaled)[0]
        probabilities = model.predict_proba(feature_array_scaled)[0]
        
        # Map prediction to label
        class_labels = {1: "Normal", 2: "Suspect", 3: "Pathological"}
        prediction_label = class_labels.get(prediction, "Unknown")
        
        # Calculate confidence and risk level
        confidence = float(np.max(probabilities))
        
        if prediction == 1:
            risk_level = "Low Risk"
        elif prediction == 2:
            risk_level = "Medium Risk - Requires Monitoring"
        else:
            risk_level = "High Risk - Immediate Attention Required"
        
        # Format probabilities
        prob_dict = {
            "Normal": float(probabilities[0]),
            "Suspect": float(probabilities[1]),
            "Pathological": float(probabilities[2])
        }
        
        return PredictionResponse(
            prediction=int(prediction),
            prediction_label=prediction_label,
            probabilities=prob_dict,
            confidence=confidence,
            risk_level=risk_level
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/batch_predict")
async def batch_predict(features_list: List[CTGFeatures]):
    """
    Predict fetal health status for multiple CTG recordings.
    
    Returns list of predictions.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        for features in features_list:
            result = await predict(features)
            results.append(result)
        
        return {
            "count": len(results),
            "predictions": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
