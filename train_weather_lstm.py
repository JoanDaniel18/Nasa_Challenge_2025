"""
Script para entrenar modelo LSTM para predicción climática
Ejecuta esto en tu máquina local (VSCode/Kali Linux)

LSTM es mejor para series temporales que Random Forest
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import glob
import os

# TensorFlow/Keras para LSTM
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    print("✅ TensorFlow importado correctamente")
except ImportError:
    print("❌ ERROR: TensorFlow no está instalado")
    print("Instala con: pip install tensorflow  (o pip3 install tensorflow)")
    print("Para Kali: pipx install tensorflow")
    exit(1)

print("="*60)
print("🧠 ENTRENAMIENTO DE MODELO LSTM")
print("Predicción: Soleado vs Nublado")
print("="*60)

# Configuración
DATA_FOLDER = "nasa_power_ecuador"
SEQUENCE_LENGTH = 24  # Usar últimas 24 horas para predecir
FEATURES = ['T2M', 'RH2M', 'WS10M', 'hour', 'month', 'day_of_year', 'is_day', 'temp_humidity_interaction']

print(f"\n📂 Cargando datos desde: {DATA_FOLDER}")

# Cargar todos los archivos CSV
all_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
print(f"   Archivos encontrados: {len(all_files)}")

if len(all_files) == 0:
    print(f"❌ ERROR: No se encontraron archivos CSV en {DATA_FOLDER}")
    exit(1)

# Leer y combinar todos los CSV
df_list = []
errors = 0
for idx, file in enumerate(all_files):
    try:
        # Intentar leer saltando líneas de comentarios
        df_temp = pd.read_csv(file, comment='#', skip_blank_lines=True)
        
        # Si el archivo está vacío o no tiene las columnas necesarias, omitir
        if len(df_temp) == 0:
            errors += 1
            continue
            
        df_list.append(df_temp)
        
        if (idx + 1) % 50 == 0:
            print(f"   Procesados: {idx + 1}/{len(all_files)} archivos")
    except Exception as e:
        # Intentar con skiprows para archivos con metadatos
        try:
            df_temp = pd.read_csv(file, skiprows=13, comment='#')
            if len(df_temp) > 0:
                df_list.append(df_temp)
            else:
                errors += 1
        except:
            errors += 1

if errors > 0:
    print(f"   ⚠️  {errors} archivos omitidos por errores de formato")

df = pd.concat(df_list, ignore_index=True)
print(f"✅ Datos cargados: {len(df):,} registros")

# Detectar nombre de columna de fecha
date_col = None
for col in ['YYYYMMDDHH', 'DATE', 'date', 'Time']:
    if col in df.columns:
        date_col = col
        break

# Si solo hay YEAR, MO, DY, HR, construir fecha
if date_col is None and all(c in df.columns for c in ['YEAR', 'MO', 'DY', 'HR']):
    print(f"   Construyendo fecha desde YEAR, MO, DY, HR...")
    df['DATE'] = pd.to_datetime(df[['YEAR', 'MO', 'DY', 'HR']].rename(
        columns={'YEAR': 'year', 'MO': 'month', 'DY': 'day', 'HR': 'hour'}
    ))
    date_col = 'DATE'
elif date_col is None:
    print(f"\n❌ ERROR: No se encontró columna de fecha")
    print(f"Columnas disponibles: {list(df.columns)}")
    exit(1)

print(f"   Usando columna de fecha: {date_col}")

# Convertir fecha a datetime si no se construyó arriba
if date_col != 'DATE':
    if date_col == 'YYYYMMDDHH':
        df['DATE'] = pd.to_datetime(df[date_col], format='%Y%m%d%H')
    else:
        df['DATE'] = pd.to_datetime(df[date_col])

# Crear features temporales
df['hour'] = df['DATE'].dt.hour
df['month'] = df['DATE'].dt.month
df['day_of_year'] = df['DATE'].dt.dayofyear
df['is_day'] = (df['hour'] >= 6) & (df['hour'] <= 18)
df['is_day'] = df['is_day'].astype(int)

# Feature de interacción
df['temp_humidity_interaction'] = df['T2M'] * df['RH2M']

# Crear variable objetivo: Soleado (1) vs Nublado (0)
# Basado en radiación solar y cobertura de nubes
df['is_sunny'] = ((df['ALLSKY_SFC_SW_DWN'] > 400) & (df['CLOUD_AMT'] < 50)).astype(int)

print(f"\n📊 Distribución de clases:")
print(f"   Soleado: {(df['is_sunny'] == 1).sum():,} ({(df['is_sunny'] == 1).sum()/len(df)*100:.2f}%)")
print(f"   Nublado: {(df['is_sunny'] == 0).sum():,} ({(df['is_sunny'] == 0).sum()/len(df)*100:.2f}%)")

# Ordenar por fecha para series temporales (antes de filtrar columnas)
sort_cols = []
if 'LAT' in df.columns:
    sort_cols.append('LAT')
if 'LON' in df.columns:
    sort_cols.append('LON')
sort_cols.append('DATE')

df = df.sort_values(sort_cols).reset_index(drop=True)
print(f"   Datos ordenados por: {', '.join(sort_cols)}")

# Seleccionar solo las features necesarias
feature_cols = FEATURES + ['is_sunny']
df = df[feature_cols].dropna()

print(f"\n🔄 Preparando secuencias temporales...")
print(f"   Longitud de secuencia: {SEQUENCE_LENGTH} horas")

# Crear secuencias para LSTM
def create_sequences(data, seq_length):
    """
    Crea secuencias de datos para LSTM
    Cada secuencia son las últimas seq_length horas
    """
    X_sequences = []
    y_labels = []
    
    # Procesar en bloques para evitar mezclar ubicaciones diferentes
    for i in range(seq_length, len(data)):
        # Tomar las últimas seq_length observaciones
        sequence = data[i-seq_length:i, :-1]  # Todas las features excepto is_sunny
        label = data[i, -1]  # is_sunny del punto actual
        
        X_sequences.append(sequence)
        y_labels.append(label)
    
    return np.array(X_sequences), np.array(y_labels)

# Convertir a numpy array
data_array = df.values

# Normalizar features (excepto is_sunny que es la última columna)
scaler = StandardScaler()
data_array[:, :-1] = scaler.fit_transform(data_array[:, :-1])

print(f"   Normalizando datos con StandardScaler...")

# Crear secuencias
X_sequences, y = create_sequences(data_array, SEQUENCE_LENGTH)

print(f"✅ Secuencias creadas: {len(X_sequences):,}")
print(f"   Shape de X: {X_sequences.shape}")
print(f"   Shape de y: {y.shape}")

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_sequences, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 División de datos:")
print(f"   Train: {len(X_train):,} secuencias")
print(f"   Test:  {len(X_test):,} secuencias")

# Construir modelo LSTM
print(f"\n🏗️  Construyendo modelo LSTM...")

model = Sequential([
    # Primera capa LSTM
    LSTM(128, return_sequences=True, input_shape=(SEQUENCE_LENGTH, len(FEATURES))),
    BatchNormalization(),
    Dropout(0.3),
    
    # Segunda capa LSTM
    LSTM(64, return_sequences=False),
    BatchNormalization(),
    Dropout(0.3),
    
    # Capas densas
    Dense(32, activation='relu'),
    Dropout(0.2),
    
    Dense(16, activation='relu'),
    
    # Capa de salida (clasificación binaria)
    Dense(1, activation='sigmoid')
])

# Compilar modelo
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\n📋 Arquitectura del modelo:")
model.summary()

# Callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

# Entrenar modelo
print(f"\n🚀 Iniciando entrenamiento...")
print(f"   Esto puede tomar varios minutos...")

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=128,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Evaluar modelo
print(f"\n📊 Evaluando modelo...")

y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*60}")
print(f"✅ RESULTADOS FINALES")
print(f"{'='*60}")
print(f"🎯 Accuracy: {accuracy*100:.2f}%")
print(f"\n📊 Reporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=['Nublado', 'Soleado']))

print(f"\n📊 Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

# Guardar modelo y scaler
print(f"\n💾 Guardando modelo y scaler...")

# Guardar modelo Keras
model.save('weather_model_lstm.h5')
print(f"   ✅ Modelo LSTM guardado: weather_model_lstm.h5")

# Guardar scaler
joblib.dump(scaler, 'weather_scaler.pkl')
print(f"   ✅ Scaler guardado: weather_scaler.pkl")

# Guardar configuración
config = {
    'sequence_length': SEQUENCE_LENGTH,
    'features': FEATURES,
    'accuracy': float(accuracy),
    'model_type': 'LSTM'
}
joblib.dump(config, 'weather_lstm_config.pkl')
print(f"   ✅ Configuración guardada: weather_lstm_config.pkl")

print(f"\n{'='*60}")
print(f"✅ ENTRENAMIENTO COMPLETADO")
print(f"{'='*60}")
print(f"\n📦 Archivos generados:")
print(f"   1. weather_model_lstm.h5 - Modelo LSTM")
print(f"   2. weather_scaler.pkl - Normalizador de datos")
print(f"   3. weather_lstm_config.pkl - Configuración")
print(f"\n💡 SIGUIENTE PASO:")
print(f"   Sube estos 3 archivos a Replit para usar el modelo LSTM")
print(f"{'='*60}")
