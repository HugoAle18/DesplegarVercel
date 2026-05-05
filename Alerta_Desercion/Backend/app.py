from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

# 1. Inicializar la app
app = FastAPI(
    title="API de Alerta Temprana Universitaria",
    description="Sistema predictivo de deserción estudiantil"
)

# 2. Configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. CARGA GLOBAL DE MODELOS ---
# Inicializamos como None para evitar el error "not defined"
mlp = None
scaler = None
imputer = None

try:
    # Intentamos cargar los archivos
    mlp = joblib.load('modelo_mlp_calibrado.pkl')
    scaler = joblib.load('scaler_fase3.pkl')
    imputer = joblib.load('imputer_fase3.pkl')
    print("✅ Todos los modelos y preprocesadores cargados correctamente")
except Exception as e:
    print(f"❌ ERROR CRÍTICO AL CARGAR MODELOS: {str(e)}")
    # No detenemos la app, pero las predicciones fallarán con un mensaje claro

# 4. Estructura de datos
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

# 5. Endpoint de Predicción
@app.post("/predecir_riesgo")
def predecir(data: EstudianteData):
    # Verificación de seguridad: ¿Cargaron los modelos?
    if mlp is None or scaler is None or imputer is None:
        raise HTTPException(
            status_code=503, 
            detail="El motor de IA no está disponible porque los archivos .pkl no cargaron correctamente. Revisa los logs de Render."
        )

    try:
        # Procesamiento de datos
        c1 = float(data.cursos_s1) if data.cursos_s1 > 0 else 1.0
        c2 = float(data.cursos_s2) if data.cursos_s2 > 0 else 1.0
        n1 = float(data.nota_s1)
        n2 = float(data.nota_s2)

        # Ingeniería de variables
        input_dict = {
            'Curricular units 1st sem (approved)': [float(data.aprobados_s1)],
            'Curricular units 1st sem (grade)': [n1],
            'Curricular units 2nd sem (approved)': [float(data.aprobados_s2)],
            'Curricular units 2nd sem (grade)': [n2],
            'Tuition fees up to date': [int(data.pago_al_dia)],
            'Debtor': [int(data.deudor)],
            'Scholarship holder': [int(data.becado)],
            'Age at enrollment': [float(data.edad)],
            'Displaced': [int(data.desplazado)],
            'Gender': [int(data.genero)],
            'GDP': [float(data.pbi)],
            'Unemployment rate': [float(data.desempleo)],
            'Inflation rate': [float(data.inflacion)],
            'aprobacion_rate_1': [float(data.aprobados_s1) / c1],
            'aprobacion_rate_2': [float(data.aprobados_s2) / c2],
            'variacion_rendimiento': [n2 - n1],
            'carga_total': [c1 + c2],
            'riesgo_financiero': [float((data.pago_al_dia == 0) + (data.deudor == 1) + (data.becado == 0))],
            'ratio_notas': [n2 / (n1 + 0.001)],
            'estres_academico': [(c1 + c2) / (float(data.edad) + 1.0)]
        }

        df_input = pd.DataFrame(input_dict)
        
        # Transformación e Inferencia
        X_transformed = imputer.transform(df_input)
        X_final = scaler.transform(X_transformed)
        proba = float(mlp.predict_proba(X_final)[0][1])
        
        if proba >= 0.70: nivel = "CRÍTICO"
        elif proba >= 0.40: nivel = "MEDIO"
        else: nivel = "BAJO"

        return {
            "probabilidad_desercion_pct": round(proba * 100, 2),
            "nivel_riesgo": nivel,
            "status": "success"
        }

    except Exception as e:
        print(f"⚠️ Error en predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en el cálculo: {str(e)}")
