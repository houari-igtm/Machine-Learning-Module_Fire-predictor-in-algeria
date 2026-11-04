from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
app = FastAPI(title='Fire prediction in algeria')

with open('../model/model.pkl','rb')as f:
    model=pickle.load(f)
with open('../model/scaler.pkl','rb')as f:
    scaler=pickle.load(f)

class Data(BaseModel):
    month: float
    Temperature: float
    RH: float
    Ws: float
    Rain: float
    FFMC: float
    DMC: float
    DC: float
    ISI: float
    BUI: float
    FWI: float


@app.get('/')
def home():
    return {"message": 'welcome to the fire predictor in algeria API','developed by':'belhamri imad eddine lahouari'}

@app.post('/predict')
def predict(data:Data):
   values = np.array([[
        data.month,
        data.Temperature,
        data.RH,
        data.Ws,
        data.Rain,
        data.FFMC,
        data.DMC,
        data.DC,
        data.ISI,
        data.BUI,
        data.FWI,
    ]])
   x_scaled=scaler.transform(values)
   y_predicted=model.predict(x_scaled)
   if y_predicted==1:
       return {'result':'there is fire!!!'}
   else:
       return {'result':'no fire'}