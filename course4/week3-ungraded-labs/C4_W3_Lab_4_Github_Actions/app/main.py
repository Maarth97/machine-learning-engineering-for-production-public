import pickle
import numpy as np
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel, conlist



app = FastAPI(title="Predicting Wine Class with batching")

# Here are some new features
# Open classifier in global scope
with open("models/wine-95.pkl", "rb") as file:
    clf = pickle.load(file)
#khuhgzi sfg dgsdgsfdg

# Add something new

class Wine(BaseModel):
    batches: List[conlist(item_type=float, min_items=13, max_items=13)]


@app.post("/predict")
def predict(wine: Wine):
    batches = wine.batches
    np_batches = np.array(batches)
    pred = clf.predict(np_batches).tolist()
    return {"Prediction": pred}
