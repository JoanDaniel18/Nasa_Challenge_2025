"""
API Demo para predicción de clima (Soleado vs Nublado)
Usa el modelo weather_model.pkl entrenado
"""
import joblib
import pandas as pd
from datetime import datetime
from typing import Optional
import numpy as np

class WeatherPredictor:
    def __init__(self, model_path='weather_model.pkl'):
        """Carga el modelo entrenado o usa predictor basado en reglas"""
        self.use_ml_model = False
        self.model = None
        self.features = None
        
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.features = model_data['features']
            self.use_ml_model = True
            print(f"✅ Modelo ML cargado exitosamente")
            print(f"   Features: {self.features}")
        except FileNotFoundError:
            print(f"⚠️  Modelo no encontrado, usando predictor basado en reglas")
            self.use_ml_model = False
    
    def get_weather_data_for_location(self, lat: float, lon: float, fecha: str, hora: int):
        """
        Simula obtener datos meteorológicos para una ubicación y hora específica
        
        En producción, esto consultaría NASA API o base de datos histórica
        Por ahora, usa valores promedio basados en la ubicación y fecha
        """
        # Convertir fecha a datetime
        date = datetime.strptime(fecha, '%Y-%m-%d')
        
        # Simular datos meteorológicos basados en patrones conocidos
        # Ecuador: clima tropical, variaciones por altitud y estación
        
        # Temperatura base (°C) - varía con latitud y hora
        temp_base = 25 - (abs(lat) * 2)  # Más frío en latitudes altas
        temp_variation = 5 * np.sin((hora - 6) * np.pi / 12)  # Ciclo diario
        temperature = temp_base + temp_variation
        
        # Humedad (%) - más alta en costa (longitudes bajas)
        humidity_base = 85 - (abs(lon + 78) * 5)  # Costa más húmeda
        humidity_variation = -10 * np.sin((hora - 6) * np.pi / 12)  # Baja al mediodía
        humidity = max(30, min(100, humidity_base + humidity_variation))
        
        # Viento (m/s) - promedio regional
        wind_speed = 1.0 + np.random.uniform(-0.3, 0.3)
        
        # Features temporales
        month = date.month
        day_of_year = date.timetuple().tm_yday
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
    
    def predict_rule_based(self, weather_data):
        """
        Predicción basada en reglas cuando no hay modelo ML
        Simula el comportamiento del modelo usando lógica simple
        """
        temp = weather_data['T2M']
        humidity = weather_data['RH2M']
        hour = weather_data['hour']
        
        # Reglas simples para predecir:
        # - Soleado si: baja humedad Y hora del día Y temperatura alta
        # - Nublado si: alta humedad O temperatura baja
        
        score = 0.5  # Base neutral
        
        # Factor temperatura (más calor = más soleado)
        if temp > 28:
            score += 0.2
        elif temp < 22:
            score -= 0.2
        
        # Factor humedad (menos humedad = más soleado)
        if humidity < 60:
            score += 0.25
        elif humidity > 85:
            score -= 0.25
        
        # Factor hora (mediodía = más soleado)
        if 10 <= hour <= 15:
            score += 0.15
        elif hour < 6 or hour > 19:
            score -= 0.2
        
        # Convertir score a predicción
        prob_sunny = max(0.1, min(0.9, score))
        prob_cloudy = 1 - prob_sunny
        
        prediction = 1 if prob_sunny > 0.5 else 0
        
        return prediction, [prob_cloudy, prob_sunny]
    
    def predict(self, lat: float, lon: float, fecha: str, hora: int):
        """
        Predice si estará soleado o nublado
        
        Args:
            lat: Latitud (-5 a 5 para Ecuador)
            lon: Longitud (-81 a -75 para Ecuador)
            fecha: Fecha en formato 'YYYY-MM-DD'
            hora: Hora del día (0-23)
        
        Returns:
            dict con predicción y probabilidades
        """
        # Validar inputs
        if not (-5 <= lat <= 5):
            raise ValueError("Latitud debe estar entre -5 y 5 (Ecuador)")
        if not (-81 <= lon <= -75):
            raise ValueError("Longitud debe estar entre -81 y -75 (Ecuador)")
        if not (0 <= hora <= 23):
            raise ValueError("Hora debe estar entre 0 y 23")
        
        # Obtener datos meteorológicos simulados
        weather_data = self.get_weather_data_for_location(lat, lon, fecha, hora)
        
        # Hacer predicción según el modo disponible
        if self.use_ml_model:
            # Usar modelo ML
            input_df = pd.DataFrame([weather_data])
            input_df = input_df[self.features]
            prediction = self.model.predict(input_df)[0]
            probabilities = self.model.predict_proba(input_df)[0]
            mode = "ML Model (Random Forest)"
        else:
            # Usar reglas
            prediction, probabilities = self.predict_rule_based(weather_data)
            mode = "Rule-Based Predictor (Demo)"
        
        # Preparar resultado
        result = {
            'ubicacion': {
                'latitud': lat,
                'longitud': lon
            },
            'fecha_hora': {
                'fecha': fecha,
                'hora': hora
            },
            'datos_meteorologicos': {
                'temperatura_c': round(weather_data['T2M'], 2),
                'humedad_pct': round(weather_data['RH2M'], 2),
                'viento_ms': round(weather_data['WS10M'], 2)
            },
            'prediccion': {
                'condicion': 'Soleado ☀️' if prediction == 1 else 'Nublado ☁️',
                'es_soleado': bool(prediction),
                'confianza_pct': round(probabilities[prediction] * 100, 2),
                'probabilidad_soleado_pct': round(probabilities[1] * 100, 2),
                'probabilidad_nublado_pct': round(probabilities[0] * 100, 2),
                'modo': mode
            }
        }
        
        return result

# Instancia global del predictor
predictor = WeatherPredictor()
