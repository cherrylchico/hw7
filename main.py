import joblib
import json
from fastapi import FastAPI, HTTPException  
from pydantic import BaseModel
from pandas import json_normalize

class Item(BaseModel):
    "Base Item for Prediction"
    value: str

app = FastAPI()

@app.post("/predict")
aysnc def predict(sample_input: Item):
    """
    Predict from Random Forest Model in Hw5 given a Json Input
    """

    try:
        model = joblib.load("model_rf.pickle")
        json_data = json.loads(sample_input)
        X_new = json_normalize(json_data)
        prediction = model.predict(X_new)
        return {"prediction": prediction[0][0]}
    except HTTPException:
        raise {"message":"model prediction failed" }
    

