# -*- coding: utf-8 -*-
"""
Random Forest Entrenamiento Local - Exportación completa para API
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

# =========================
# CARGAR DATASET LOCAL
# =========================
def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
    df = pd.read_csv(path)
    print(f"✅ Archivo cargado: {path}")
    print(f"   Filas: {len(df)}, Columnas: {len(df.columns)}")
    return df

# =========================
# CREAR FEATURES
# =========================
def create_features(df):
    """Convierte datetime en features temporales y prepara X, y"""
    print("\n🔧 Creando features...")

    # Asegurar formato datetime
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Features temporales
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['day_of_year'] = df['datetime'].dt.dayofyear

    # Inputs
    X = df[['lat', 'lon', 'hour', 'month', 'day_of_year']]

    # Outputs (todas las variables dependientes, incluida PoP)
    y = df[['T2M', 'RH2M', 'WS10M', 'CLOUD_AMT', 'ALLSKY_SFC_SW_DWN', 'PoP']]

    return X, y

# =========================
# ENTRENAR MODELO
# =========================
def train_model(X, y):
    print("\n🤖 Entrenando modelo Random Forest multisalida...")

    # Crear y ajustar scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Modelo de regresión multisalida con verbose
    model = MultiOutputRegressor(RandomForestRegressor(
        n_estimators=50,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
        verbose=2
    ))

    print("   ⏳ Entrenando...")
    model.fit(X_train, y_train)

    # Evaluación
    y_pred = model.predict(X_test)

    print("\n📊 Resultados del modelo:")
    metrics = {}
    for i, col in enumerate(y.columns):
        mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
        r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
        print(f"   • {col}: MSE={mse:.3f}, R²={r2:.3f}")
        metrics[col] = {'mse': mse, 'r2': r2}

    return model, scaler, metrics

# =========================
# GUARDAR ARCHIVOS PARA API
# =========================
def save_model_files(model, scaler, X, y, metrics):
    """Guarda modelo, scaler y configuración para la API"""
    
    print("\n💾 Guardando archivos para la API...")
    
    # 1. Guardar modelo
    model_file = 'weather_model.pkl'
    joblib.dump(model, model_file)
    print(f"   ✅ Modelo guardado: {model_file}")
    
    # 2. Guardar scaler
    scaler_file = 'weather_scaler.pkl'
    joblib.dump(scaler, scaler_file)
    print(f"   ✅ Scaler guardado: {scaler_file}")
    
    # 3. Guardar configuración
    config = {
        'model_type': 'Random Forest MultiOutput',
        'features': list(X.columns),
        'output_variables': list(y.columns),
        'n_features': len(X.columns),
        'n_outputs': len(y.columns),
        'metrics': metrics,
        'test_size': 0.2,
        'random_state': 42
    }
    
    config_file = 'weather_config.pkl'
    joblib.dump(config, config_file)
    print(f"   ✅ Configuración guardada: {config_file}")
    
    print("\n📦 Archivos generados para la API:")
    print(f"   1. {model_file}")
    print(f"   2. {scaler_file}")
    print(f"   3. {config_file}")
    
    print("\n💡 Copia estos 3 archivos a la carpeta 'models/' de tu API")

# =========================
# MAIN
# =========================
def main():
    print("="*60)
    print("🌤️  ENTRENAMIENTO DE MODELO CLIMÁTICO (Regresión multisalida)")
    print("="*60)

    # Ruta absoluta al archivo CSV
    path = "/home/daniel/Escritorio/Hackaton/df_all_con_PoP.csv"
    
    df = load_data(path)
    X, y = create_features(df)
    model, scaler, metrics = train_model(X, y)
    save_model_files(model, scaler, X, y, metrics)

    print("\n" + "="*60)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("="*60)
    print("\n🚀 Siguiente paso:")
    print("   Copia los 3 archivos .pkl a la carpeta models/ de tu API FastAPI")

# =========================
# EJECUCIÓN
# =========================
if __name__ == "__main__":
    main()