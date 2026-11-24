# code - api 
# fastapi-rest api , uvicorn-8000, basemodel-structure and validation
# @ - annotation - set of instructions  "/" - router  
# github deploy = git action and docker // render 

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# Enable CORS for live frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# uvicorn main:app --reload
model_path = "logisticmodel.pkl" if os.path.exists("logisticmodel.pkl") else "logisticrmodel.pkl"
model = joblib.load(model_path)

class datascience(BaseModel):
    StudyHours: float

@app.get("/")
def test():
    return {"msg": "uvicorn is running"}

@app.post("/predict")
def datainsert(ds: datascience):
    input_data = pd.DataFrame([[ds.StudyHours]], columns=["StudyHours"])
    prediction = int(model.predict(input_data)[0])
    result = "Pass" if prediction == 1 else "Fail"
    proba = model.predict_proba(input_data)[0]

    return {
        "StudyHours": ds.StudyHours,
        "prediction": prediction,
        "result": result,
        "probability": round(float(proba[1]), 4)
    }
