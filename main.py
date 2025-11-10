import joblib
import json
import numpy as np
from fastapi import FastAPI, HTTPException  
from pandas import json_normalize

app = FastAPI()

@app.post("/predict")
async def predict(sample_input: dict):
    """Predict endpoint.

    Accepts a JSON object representing a single record in json format.

    Returns: {"prediction": <value>} where <value> is the model's output for the record.
    """

    try:
        # Load model
        model = joblib.load("model_rf.pickle")

        #Convert input into a Dataframe
        X_new = json_normalize([sample_input])

        # Make prediction, note that input is a dataframe
        prediction = model.predict(X_new)

        # Get the first (and should be only) prediction
        pred_value = prediction.iloc[0,0]
        return {"prediction": pred_value}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")