import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, create_model
from pandas import json_normalize
from typing import Dict

app = FastAPI(title="ML Classification API", version="1.0")

# Load model once at startup
try:
    model = joblib.load("model_rf.pickle")
    feature_cols = model._feature_cols
    print("Model loaded successfully")
    print(f"Features expected: {feature_cols}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    feature_cols = []



# Dynamically create Pydantic model based on model's features
def create_input_schema(features):
    """Creates a Pydantic model with fields matching model features"""
    fields = {
        feature: (float, Field(..., description=f"Value for {feature}"))
        for feature in features
    }
    return create_model('PredictionInput', **fields)


# Create input schema if model loaded successfully
if feature_cols:
    PredictionInput = create_input_schema(feature_cols)
else:
    # Fallback generic model
    class PredictionInput(BaseModel):
        pass


@app.post("/predict")
async def predict(sample_input: PredictionInput):
    """Predict endpoint.
    
    Accepts a JSON object with all required features.
    Automatically validates that all features are present and are numeric.
    
    Returns: 
        {
            "prediction": <probability for class 1>,
            "probabilities": {
                "class_0": <prob>,
                "class_1": <prob>
            },
            "predicted_class": <0 or 1>
        }
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert Pydantic model to dict, then to DataFrame
        input_dict = sample_input.model_dump()
        X_new = json_normalize([input_dict])
        
        # Make prediction (returns DataFrame with probabilities)
        prediction = model.predict(X_new)
        
        # Get the first (and should be only) prediction
        pred_value = prediction.iloc[0,1]

        return {"prediction": pred_value}
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction error: {str(e)}"
        )