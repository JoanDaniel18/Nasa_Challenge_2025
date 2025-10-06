"""
Predictor usando modelo LSTM
Compatible con el sistema actual (weather_demo_api.py)
"""
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta

try:
    from tensorflow.keras.models import load_model
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("⚠️  TensorFlow no disponible, instalar con: pip install tensorflow")

class LSTMWeatherPredictor:
    """
    Predictor de clima usando LSTM
    Emula la interfaz de sklearn para compatibilidad
    """
    
    def __init__(self, model_path='weather_model_lstm.h5', 
                 scaler_path='weather_scaler.pkl',
                 config_path='weather_lstm_config.pkl'):
        """
        Carga modelo LSTM, scaler y configuración
        """
        if not KERAS_AVAILABLE:
            raise ImportError("TensorFlow no está disponible. Instala con: pip install tensorflow")
        
        try:
            # Cargar modelo LSTM
            self.model = load_model(model_path)
            print(f"✅ Modelo LSTM cargado: {model_path}")
            
            # Cargar scaler
            self.scaler = joblib.load(scaler_path)
            print(f"✅ Scaler cargado: {scaler_path}")
            
            # Cargar configuración
            self.config = joblib.load(config_path)
            self.sequence_length = self.config['sequence_length']
            self.features = self.config['features']
            
            print(f"✅ Configuración cargada")
            print(f"   Secuencia: {self.sequence_length} timesteps")
            print(f"   Features: {len(self.features)}")
            
            # Estadísticas para simular datos históricos
            self.feature_means = {
                'T2M': 25.0,
                'RH2M': 75.0,
                'WS10M': 1.5
            }
            
        except Exception as e:
            raise Exception(f"Error cargando modelo LSTM: {e}")
    
    def create_sequence_from_point(self, weather_data, sequence_length=24):
        """
        Crea una secuencia temporal simulada para un punto individual
        LSTM necesita secuencias, así que simulamos las últimas horas
        """
        # Crear secuencia repitiendo el punto con variaciones pequeñas
        sequence = []
        
        for i in range(sequence_length):
            # Simular datos de horas anteriores con variación
            point = weather_data.copy()
            
            # Ajustar hora hacia atrás
            hours_back = sequence_length - i - 1
            point['hour'] = (point['hour'] - hours_back) % 24
            point['is_day'] = 1 if 6 <= point['hour'] <= 18 else 0
            
            # Pequeñas variaciones en temperatura y humedad
            noise_factor = np.random.normal(0, 0.1)
            point['T2M'] = point['T2M'] * (1 + noise_factor * 0.05)
            point['RH2M'] = point['RH2M'] * (1 + noise_factor * 0.03)
            
            # Recalcular interacción
            point['temp_humidity_interaction'] = point['T2M'] * point['RH2M']
            
            sequence.append([point[f] for f in self.features])
        
        return np.array(sequence)
    
    def predict(self, X):
        """
        Hace predicción (compatible con sklearn)
        X: DataFrame con las features
        """
        if isinstance(X, pd.DataFrame):
            X_dict = X.iloc[0].to_dict()
        else:
            X_dict = X
        
        # Crear secuencia
        sequence = self.create_sequence_from_point(X_dict, self.sequence_length)
        
        # Normalizar
        sequence_scaled = self.scaler.transform(sequence)
        
        # Reshape para LSTM: (1, sequence_length, n_features)
        sequence_scaled = sequence_scaled.reshape(1, self.sequence_length, len(self.features))
        
        # Predecir
        prob = self.model.predict(sequence_scaled, verbose=0)[0][0]
        
        # Convertir a clase (0 o 1)
        prediction = 1 if prob > 0.5 else 0
        
        return np.array([prediction])
    
    def predict_proba(self, X):
        """
        Retorna probabilidades (compatible con sklearn)
        """
        if isinstance(X, pd.DataFrame):
            X_dict = X.iloc[0].to_dict()
        else:
            X_dict = X
        
        # Crear secuencia
        sequence = self.create_sequence_from_point(X_dict, self.sequence_length)
        
        # Normalizar
        sequence_scaled = self.scaler.transform(sequence)
        
        # Reshape para LSTM
        sequence_scaled = sequence_scaled.reshape(1, self.sequence_length, len(self.features))
        
        # Predecir probabilidad
        prob_sunny = self.model.predict(sequence_scaled, verbose=0)[0][0]
        prob_cloudy = 1 - prob_sunny
        
        # Formato sklearn: [[prob_clase0, prob_clase1]]
        return np.array([[prob_cloudy, prob_sunny]])

# Función de ayuda para testear
if __name__ == "__main__":
    print("="*60)
    print("🧪 TEST: Predictor LSTM")
    print("="*60)
    
    try:
        # Cargar modelo
        predictor = LSTMWeatherPredictor()
        
        # Datos de prueba (Quito, mediodía)
        test_data = {
            'T2M': 25.0,
            'RH2M': 70.0,
            'WS10M': 1.2,
            'hour': 12,
            'month': 3,
            'day_of_year': 75,
            'is_day': 1,
            'temp_humidity_interaction': 25.0 * 70.0
        }
        
        print("\n📍 Datos de entrada:")
        for key, value in test_data.items():
            print(f"   {key}: {value}")
        
        # Convertir a DataFrame
        df_test = pd.DataFrame([test_data])
        
        # Predecir
        print("\n🔮 Haciendo predicción...")
        prediction = predictor.predict(df_test)
        probabilities = predictor.predict_proba(df_test)
        
        print(f"\n✅ RESULTADO:")
        print(f"   Predicción: {'Soleado ☀️' if prediction[0] == 1 else 'Nublado ☁️'}")
        print(f"   Prob. Nublado: {probabilities[0][0]*100:.2f}%")
        print(f"   Prob. Soleado: {probabilities[0][1]*100:.2f}%")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Asegúrate de tener los archivos:")
        print("  - weather_model_lstm.h5")
        print("  - weather_scaler.pkl")
        print("  - weather_lstm_config.pkl")
