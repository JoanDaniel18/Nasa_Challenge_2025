"""
BACKEND FASTAPI - PREDICCIÓN CLIMA ECUADOR (LSTM)
==================================================
API REST para servir predicciones de clima usando modelo LSTM

Archivos requeridos:
- weather_model_lstm.h5 (modelo LSTM entrenado)
- weather_scaler.pkl (normalizador de datos)
- weather_lstm_config.pkl (configuración del modelo)

Ejecutar: python lstm_weather_backend.py
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import joblib
import numpy as np
import uvicorn
import sys

# TensorFlow/Keras
try:
    from tensorflow.keras.models import load_model
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')  # Usar CPU
    print("✅ TensorFlow importado correctamente")
except ImportError:
    print("❌ ERROR: TensorFlow no está instalado")
    print("Instala con: pip install tensorflow")
    sys.exit(1)

# ============================================================
# CONFIGURACIÓN DE LA API
# ============================================================

app = FastAPI(
    title="Ecuador Weather API - LSTM",
    description="Predicción de clima (Soleado/Nublado) usando modelo LSTM",
    version="1.0.0"
)

# CORS para permitir requests desde Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# VARIABLES GLOBALES DEL MODELO
# ============================================================

lstm_model = None
scaler = None
config = None
features = None
sequence_length = None

# ============================================================
# MODELOS DE DATOS (REQUEST/RESPONSE)
# ============================================================

class PredictionRequest(BaseModel):
    latitude: float
    longitude: float
    datetime: str  # ISO 8601: "2025-10-05T14:30:00"

class PredictionResponse(BaseModel):
    ubicacion: dict
    fecha_hora: dict
    datos_meteorologicos: dict
    prediccion: dict
    modelo_info: dict

# ============================================================
# CARGA DEL MODELO AL INICIAR
# ============================================================

@app.on_event("startup")
async def load_lstm_model():
    """Carga el modelo LSTM y sus componentes al iniciar el servidor"""
    global lstm_model, scaler, config, features, sequence_length
    
    print("\n" + "="*60)
    print("🧠 CARGANDO MODELO LSTM")
    print("="*60)
    
    try:
        # 1. Cargar modelo LSTM
        print("\n📦 Cargando modelo LSTM...")
        lstm_model = load_model('weather_model_lstm.h5', compile=False)
        lstm_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        print("✅ Modelo LSTM cargado: weather_model_lstm.h5")
        
        # 2. Cargar scaler
        print("📦 Cargando scaler...")
        scaler = joblib.load('weather_scaler.pkl')
        print("✅ Scaler cargado: weather_scaler.pkl")
        
        # 3. Cargar configuración
        print("📦 Cargando configuración...")
        config = joblib.load('weather_lstm_config.pkl')
        features = config['features']
        sequence_length = config['sequence_length']
        print("✅ Configuración cargada: weather_lstm_config.pkl")
        
        # Mostrar info del modelo
        print(f"\n📊 INFORMACIÓN DEL MODELO:")
        print(f"   Tipo: {config.get('model_type', 'LSTM')}")
        print(f"   Accuracy: {config.get('accuracy', 'N/A')}")
        print(f"   Secuencia: {sequence_length} timesteps")
        print(f"   Features: {len(features)}")
        print(f"   Lista de features: {', '.join(features)}")
        print("="*60 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: Archivo no encontrado")
        print(f"   {e}")
        print("\n📦 Archivos requeridos:")
        print("   1. weather_model_lstm.h5")
        print("   2. weather_scaler.pkl")
        print("   3. weather_lstm_config.pkl")
        print("\n💡 Coloca estos archivos en la misma carpeta que este script")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ ERROR al cargar modelo: {e}")
        sys.exit(1)

# ============================================================
# FUNCIONES DE PREDICCIÓN
# ============================================================

def get_weather_data(lat: float, lon: float, dt: datetime) -> dict:
    """
    Simula datos meteorológicos basados en ubicación y hora
    En producción, esto consultaría una base de datos real o API
    """
    hora = dt.hour
    
    # Temperatura (°C) - varía con latitud y hora
    temp_base = 25 - (abs(lat) * 2)
    temp_variation = 5 * np.sin((hora - 6) * np.pi / 12)
    temperature = temp_base + temp_variation
    
    # Humedad (%) - varía con longitud y hora
    humidity_base = 85 - (abs(lon + 78) * 5)
    humidity_variation = -10 * np.sin((hora - 6) * np.pi / 12)
    humidity = max(30, min(100, humidity_base + humidity_variation))
    
    # Viento (m/s)
    wind_speed = 1.0 + np.random.uniform(-0.3, 0.3)
    
    # Features temporales
    month = dt.month
    day_of_year = dt.timetuple().tm_yday
    is_day = 1 if 6 <= hora <= 18 else 0
    
    # Interacción temperatura-humedad
    temp_humidity_interaction = temperature * humidity
    
    return {
        'T2M': temperature,
        'RH2M': humidity,
        'WS10M': wind_speed,
        'hour': hora,
        'month': month,
        'day_of_year': day_of_year,
        'is_day': is_day,
        'temp_humidity_interaction': temp_humidity_interaction
    }

def create_lstm_sequence(weather_data: dict) -> np.ndarray:
    """
    Crea una secuencia temporal para LSTM
    LSTM necesita las últimas N horas, así que las simulamos
    """
    sequence = []
    
    for i in range(sequence_length):
        point = weather_data.copy()
        
        # Ajustar hora hacia atrás
        hours_back = sequence_length - i - 1
        point['hour'] = (point['hour'] - hours_back) % 24
        point['is_day'] = 1 if 6 <= point['hour'] <= 18 else 0
        
        # Variaciones pequeñas en temperatura y humedad
        noise = np.random.normal(0, 0.1)
        point['T2M'] = point['T2M'] * (1 + noise * 0.05)
        point['RH2M'] = point['RH2M'] * (1 + noise * 0.03)
        point['temp_humidity_interaction'] = point['T2M'] * point['RH2M']
        
        # Agregar features en el orden correcto
        sequence.append([point[f] for f in features])
    
    return np.array(sequence)

def predict_with_lstm(weather_data: dict) -> tuple:
    """
    Hace predicción usando modelo LSTM
    Returns: (prediction, probabilities)
    """
    # Crear secuencia
    sequence = create_lstm_sequence(weather_data)
    
    # Normalizar con el scaler
    sequence_scaled = scaler.transform(sequence)
    
    # Reshape para LSTM: (1, sequence_length, n_features)
    X = sequence_scaled.reshape(1, sequence_length, len(features))
    
    # Predecir
    prob_sunny = float(lstm_model.predict(X, verbose=0)[0][0])
    prob_cloudy = 1.0 - prob_sunny
    
    # Clasificación
    prediction = 1 if prob_sunny > 0.5 else 0
    
    return prediction, [prob_cloudy, prob_sunny]

# ============================================================
# ENDPOINTS DE LA API
# ============================================================

@app.get("/")
async def root():
    """Endpoint raíz - Info básica de la API"""
    return {
        "app": "Ecuador Weather API - LSTM",
        "version": "1.0.0",
        "model_type": "LSTM",
        "status": "running",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "model_info": "/model-info",
            "predict": "/predict"
        }
    }

@app.get("/health")
async def health():
    """Health check - Estado del servidor y modelo"""
    return {
        "status": "healthy",
        "model_type": "LSTM",
        "model_loaded": lstm_model is not None,
        "sequence_length": sequence_length,
        "features_count": len(features) if features else 0
    }

@app.get("/model-info")
async def model_info():
    """Información detallada del modelo"""
    if lstm_model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    return {
        "model_type": config.get('model_type', 'LSTM'),
        "sequence_length": sequence_length,
        "features": features,
        "accuracy": config.get('accuracy', 'N/A'),
        "input_shape": f"({sequence_length}, {len(features)})"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    ENDPOINT PRINCIPAL DE PREDICCIÓN
    
    Request JSON:
    {
        "latitude": -0.1807,
        "longitude": -78.4678,
        "datetime": "2025-10-05T14:30:00"
    }
    
    Response JSON:
    {
        "prediccion": {
            "condicion": "Soleado",
            "condicion_emoji": "☀️ Soleado",
            "codigo": 1,
            "confianza_pct": 87.45,
            ...
        }
    }
    """
    # Validar que el modelo esté cargado
    if lstm_model is None:
        raise HTTPException(status_code=503, detail="Modelo LSTM no disponible")
    
    # Validar coordenadas para Ecuador
    if not (-5 <= request.latitude <= 5):
        raise HTTPException(
            status_code=400,
            detail="Latitud debe estar entre -5 y 5 (rango de Ecuador)"
        )
    
    if not (-81 <= request.longitude <= -75):
        raise HTTPException(
            status_code=400,
            detail="Longitud debe estar entre -81 y -75 (rango de Ecuador)"
        )
    
    # Parsear fecha/hora
    try:
        dt = datetime.fromisoformat(request.datetime.replace('Z', '+00:00'))
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Formato de fecha inválido. Usar ISO 8601 (ej: 2025-10-05T14:30:00)"
        )
    
    # Obtener datos meteorológicos simulados
    weather_data = get_weather_data(request.latitude, request.longitude, dt)
    
    # Hacer predicción con LSTM
    prediction, probabilities = predict_with_lstm(weather_data)
    
    # Preparar respuesta
    response = {
        'ubicacion': {
            'latitud': request.latitude,
            'longitud': request.longitude
        },
        'fecha_hora': {
            'fecha': dt.strftime('%Y-%m-%d'),
            'hora': dt.hour,
            'datetime_iso': request.datetime,
            'hora_formato': dt.strftime('%I:%M %p')
        },
        'datos_meteorologicos': {
            'temperatura_c': round(weather_data['T2M'], 2),
            'humedad_pct': round(weather_data['RH2M'], 2),
            'viento_ms': round(weather_data['WS10M'], 2)
        },
        'prediccion': {
            'condicion': 'Soleado' if prediction == 1 else 'Nublado',
            'condicion_emoji': '☀️ Soleado' if prediction == 1 else '☁️ Nublado',
            'es_soleado': bool(prediction),
            'codigo': int(prediction),  # 0=nublado, 1=soleado
            'confianza_pct': round(probabilities[prediction] * 100, 2),
            'probabilidad_soleado_pct': round(probabilities[1] * 100, 2),
            'probabilidad_nublado_pct': round(probabilities[0] * 100, 2)
        },
        'modelo_info': {
            'tipo': 'LSTM',
            'secuencia_timesteps': sequence_length,
            'features_usadas': len(features),
            'accuracy_entrenamiento': config.get('accuracy', 'N/A')
        }
    }
    
    return response

# ============================================================
# EJECUTAR SERVIDOR
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🌤️  ECUADOR WEATHER API - LSTM")
    print("="*60)
    print("\n🚀 Iniciando servidor FastAPI...")
    print("📍 API URL: http://localhost:8000")
    print("📖 Documentación interactiva: http://localhost:8000/docs")
    print("🔬 Redoc: http://localhost:8000/redoc")
    print("\n⏳ Cargando modelo LSTM...\n")
    
    # Ejecutar servidor
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
