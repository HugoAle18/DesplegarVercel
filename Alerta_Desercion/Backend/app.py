from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# 1. Inicializar la app
app = FastAPI(
    title="API de Alerta Temprana Universitaria",
    description="Sistema predictivo de deserción estudiantil"
)

# 2. Configuración CORS (Obligatorio para conectar con Vercel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Cargar los modelos al arrancar
try:
    mlp = joblib.load('modelo_mlp_calibrado.pkl')
    scaler = joblib.load('scaler_fase3.pkl')
    imputer = joblib.load('imputer_fase3.pkl')
    print("✅ Modelos cargados correctamente")
except Exception as e:
    print(f"❌ Error al cargar los modelos: {e}")

# 4. Definir la estructura de entrada
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

# 5. Endpoint de Predicción (Con el modelo real)
@app.post("/predecir_riesgo")
def predecir(data: EstudianteData):
    try:
        # Calcular variables derivadas (Tu ingeniería de características)
        aprobacion_rate_1 = data.aprobados_s1 / data.cursos_s1 if data.cursos_s1 > 0 else 0
        aprobacion_rate_2 = data.aprobados_s2 / data.cursos_s2 if data.cursos_s2 > 0 else 0
        variacion_rendimiento = data.nota_s2 - data.nota_s1
        carga_total = data.cursos_s1 + data.cursos_s2
        riesgo_financiero = (data.pago_al_dia == 0) + (data.deudor == 1) + (data.becado == 0)
        ratio_notas = data.nota_s2 / (data.nota_s1 + 1e-5)
        estres_academico = carga_total / (data.edad + 1)

        # Empaquetar todo como lo espera el modelo
        input_dict = {
            'Curricular units 1st sem (approved)': [data.aprobados_s1],
            'Curricular units 1st sem (grade)': [data.nota_s1],
            'Curricular units 2nd sem (approved)': [data.aprobados_s2],
            'Curricular units 2nd sem (grade)': [data.nota_s2],
            'Tuition fees up to date': [data.pago_al_dia],
            'Debtor': [data.deudor],
            'Scholarship holder': [data.becado],
            'Age at enrollment': [data.edad],
            'Displaced': [data.desplazado],
            'Gender': [data.genero],
            'GDP': [data.pbi],
            'Unemployment rate': [data.desempleo],
            'Inflation rate': [data.inflacion],
            'aprobacion_rate_1': [aprobacion_rate_1],
            'aprobacion_rate_2': [aprobacion_rate_2],
            'variacion_rendimiento': [variacion_rendimiento],
            'carga_total': [carga_total],
            'riesgo_financiero': [riesgo_financiero],
            'ratio_notas': [ratio_notas],
            'estres_academico': [estres_academico]
        }

        # Convertir y preprocesar
        df_input = pd.DataFrame(input_dict)
        features_order = list(input_dict.keys())
        df_input = df_input[features_order]

        X_input = imputer.transform(df_input)
        X_input = scaler.transform(X_input)

        # Predecir usando el MLP
        proba = mlp.predict_proba(X_input)[0][1]
        
        # Lógica de niveles
        if proba >= 0.70:
            nivel = "CRÍTICO"
        elif proba >= 0.40:
            nivel = "MEDIO"
        else:
            nivel = "BAJO"

        # DEVOLVER LA ESTRUCTURA EXACTA QUE ESPERA TU FRONTEND
        return {
            "probabilidad_desercion_pct": round(proba * 100, 2),
            "alerta": int(proba >= 0.40),
            "nivel_riesgo": nivel
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
