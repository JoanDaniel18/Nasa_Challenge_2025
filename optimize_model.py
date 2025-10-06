"""
Script para optimizar y reducir el tamaño del modelo
Ejecuta esto en tu máquina local (VSCode/Kali)
"""
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Cargar el modelo original
print("Cargando modelo original...")
model = joblib.load('weather_model.pkl')
print(f"Tamaño original: {round(joblib.dump(model, 'temp.pkl')[1] / (1024*1024), 2)} MB")

# Opción 1: Reducir número de árboles (de 200 a 50-100)
print("\n=== Opción 1: Reducir árboles ===")
# Si tu modelo tiene muchos árboles, reduce a 50-100
if hasattr(model, 'n_estimators'):
    print(f"Árboles originales: {model.n_estimators}")
    # Mantener solo los primeros 50 árboles
    model.estimators_ = model.estimators_[:50]
    model.n_estimators = 50
    joblib.dump(model, 'weather_model_optimized.pkl', compress=3)
    import os
    size_mb = os.path.getsize('weather_model_optimized.pkl') / (1024*1024)
    print(f"Nuevo tamaño con 50 árboles: {round(size_mb, 2)} MB")

# Opción 2: Comprimir el modelo
print("\n=== Opción 2: Máxima compresión ===")
joblib.dump(model, 'weather_model_compressed.pkl', compress=9)
import os
size_mb = os.path.getsize('weather_model_compressed.pkl') / (1024*1024)
print(f"Tamaño con compresión máxima: {round(size_mb, 2)} MB")

# Opción 3: Combinar ambas (recomendado)
print("\n=== Opción 3: Reducir + Comprimir (RECOMENDADO) ===")
if hasattr(model, 'estimators_'):
    model_light = RandomForestClassifier(
        n_estimators=50,
        max_depth=model.max_depth if hasattr(model, 'max_depth') else 10,
        random_state=42
    )
    # Copiar solo los parámetros esenciales
    model_light.estimators_ = model.estimators_[:50]
    model_light.n_estimators = 50
    model_light.classes_ = model.classes_
    model_light.n_classes_ = model.n_classes_
    model_light.n_features_in_ = model.n_features_in_
    if hasattr(model, 'feature_names_in_'):
        model_light.feature_names_in_ = model.feature_names_in_
    
    joblib.dump(model_light, 'weather_model_light.pkl', compress=9)
    size_mb = os.path.getsize('weather_model_light.pkl') / (1024*1024)
    print(f"Tamaño final optimizado: {round(size_mb, 2)} MB")
    print("\n✅ Usa 'weather_model_light.pkl' - debería ser < 50MB")

print("\n" + "="*50)
print("SIGUIENTE PASO:")
print("1. Renombra 'weather_model_light.pkl' a 'weather_model.pkl'")
print("2. Sube ese archivo a Replit")
print("="*50)
