from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # <-- NUEVO
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(
    title="API de Alerta Temprana Universitaria",
    description="Sistema predictivo de deserción estudiantil"
)

# 👇 === NUEVO BLOQUE CORS === 👇
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite que cualquier web se conecte (ideal para pruebas)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 1. Cargar el modelo y preprocesadores al iniciar el servidor
mlp = joblib.load('modelo_mlp_calibrado.pkl')
scaler = joblib.load('scaler_fase3.pkl')
imputer = joblib.load('imputer_fase3.pkl')

# 2. Definir la estructura de datos que enviará la web
class EstudianteData(BaseModel):
    aprobados_s1: float
    nota_s1: float
    aprobados_s2: float
    nota_s2: float
    edad: float
    genero: int
    desplazado: int
    pago_al_dia: int
    deudor: int
    becado: int
    pbi: float
    desempleo: float
    inflacion: float
    cursos_s1: float
    cursos_s2: float

@app.post("/predecir_riesgo")
def predecir(data: EstudianteData):
    # Aquí recreas la lógica de tus "Features Derivadas"
    # Transformas a DataFrame -> Imputas -> Escalas -> Predices
    # ... (tu código de la Fase 10 adaptado) ...
    
    # Retornas un JSON a la web
    return {"probabilidad_desercion": 0.85, "nivel_riesgo": "CRÍTICO"}