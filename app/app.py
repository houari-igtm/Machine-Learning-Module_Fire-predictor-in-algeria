from fastapi import FastAPI
from pydantic import BaseModel

app=FastAPI(title='Fire prediction in algeria')


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

