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

# 2. Configuración CORS (Permite la conexión desde tu HTML local o Vercel)
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

# 4. Definir la estructura de entrada de datos
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

# 5. Endpoint de Predicción con Escudo Anti-Errores
@app.post("/predecir_riesgo")
def predecir(data: EstudianteData):
    try:
        # --- PROCESAMIENTO SEGURO DE DATOS ---
        # Forzamos conversión a float y evitamos divisiones por cero
        c1 = float(data.cursos_s1) if data.cursos_s1 > 0 else 1.0
        c2 = float(data.cursos_s2) if data.cursos_s2 > 0 else 1.0
        n1 = float(data.nota_s1)
        n2 = float(data.nota_s2)

        # Ingeniería de variables (Igual a como entrenaste el modelo)
        aprobacion_rate_1 = float(data.aprobados_s1) / c1
        aprobacion_rate_2 = float(data.aprobados_s2) / c2
        variacion_rendimiento = n2 - n1
        carga_total = c1 + c2
        riesgo_financiero = float((data.pago_al_dia == 0) + (data.deudor == 1) + (data.becado == 0))
        ratio_notas = n2 / (n1 + 0.001)
        estres_academico = carga_total / (float(data.edad) + 1.0)

        # Crear diccionario con las 20 columnas exactas requeridas
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
            'aprobacion_rate_1': [aprobacion_rate_1],
            'aprobacion_rate_2': [aprobacion_rate_2],
            'variacion_rendimiento': [variacion_rendimiento],
            'carga_total': [carga_total],
            'riesgo_financiero': [riesgo_financiero],
            'ratio_notas': [ratio_notas],
            'estres_academico': [estres_academico]
        }

        # Convertir a DataFrame manteniendo el orden de las columnas
        df_input = pd.DataFrame(input_dict)
        
        # Aplicar Transformaciones (Imputer -> Scaler)
        X_transformed = imputer.transform(df_input)
        X_final = scaler.transform(X_transformed)

        # Ejecutar Inferencia con el MLP
        proba = float(mlp.predict_proba(X_final)[0][1])
        
        # Definición de niveles de alerta
        if proba >= 0.70:
            nivel = "CRÍTICO"
        elif proba >= 0.40:
            nivel = "MEDIO"
        else:
            nivel = "BAJO"

        # Respuesta final para el Frontend
        return {
            "probabilidad_desercion_pct": round(proba * 100, 2),
            "nivel_riesgo": nivel,
            "status": "success"
        }

    except Exception as e:
        # Log interno para depuración en Render
        print(f"⚠️ Error en predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno del modelo: {str(e)}")
