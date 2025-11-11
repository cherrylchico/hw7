import joblib

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

        #Model requires dataframe input so convert json to dataframe
        X_new = json_normalize([sample_input])

        #Make prediction, output is a dataframe where first row is the prediction
        #first column prediction for 0
        #second columb prediction for 1
        prediction = model.predict(X_new)

        # Get the first (and should be only) prediction
        pred_value = prediction.iloc[0,1]
        return {"prediction": pred_value}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    

