# 🌎 It Will Rain on My Parade?  
### Proyecto presentado en el *NASA Space Apps Challenge Hackathon 2025*

Desarrollado por el *Team: Weather_Forecast*

👩‍💻 *Integrantes:*
- Daheny Lopez  
- Jhony Peñaherrera  
- Silvia García  
- Lander Lliguicota  
- Daniel Rivas  
- Mishell Guano

---

## 📋 Descripción del Proyecto

Sistema de predicción climática para Ecuador que utiliza un modelo **Random Forest MultiOutput** entrenado con datos históricos de NASA POWER. El sistema predice 6 variables climáticas simultáneamente para cualquier ubicación dentro del territorio ecuatoriano.

### 🎯 Variables Predichas
- **T2M**: Temperatura a 2 metros (°C)
- **RH2M**: Humedad relativa a 2 metros (%)
- **WS10M**: Velocidad del viento a 10 metros (m/s)
- **CLOUD_AMT**: Cantidad de nubosidad (%)
- **ALLSKY_SFC_SW_DWN**: Radiación solar descendente (W/m²)
- **PoP**: Probabilidad de precipitación (%)


---

## 🧪 Pruebas del Modelo - Aplicación Streamlit

Para validar el funcionamiento del modelo Random Forest, desarrollamos una **aplicación demo en Streamlit** que permite:

- Seleccionar coordenadas dentro de Ecuador mediante un mapa interactivo
- Elegir fecha y hora específicas para la predicción
- Visualizar las 6 variables climáticas predichas en tiempo real
- Verificar la precisión del modelo con datos reales

### 📊 Rendimiento del Modelo
El modelo Random Forest MultiOutput fue entrenado con un dataset de **2+ millones de registros** históricos y alcanzó los siguientes scores R²:

- Temperatura (T2M): **0.95**
- Humedad (RH2M): **0.89**
- Velocidad del viento (WS10M): **0.72**
- Nubosidad (CLOUD_AMT): **0.68**
- Radiación solar (ALLSKY_SFC_SW_DWN): **0.91**
- Probabilidad de precipitación (PoP): **0.45**

### 🎬 Demo en Streamlit
La aplicación Streamlit (`weather_demo_app.py`) sirvió como entorno de pruebas para validar:
- Carga correcta de los modelos entrenados
- Procesamiento de coordenadas geográficas
- Escalado de features con StandardScaler
- Predicciones en tiempo real
- Visualización interactiva de resultados

---

## ⚠️ Archivos del Modelo (.pkl)

Debido al tamaño considerable de los archivos del modelo entrenado (>100MB), **no fue posible subirlos directamente a GitHub**. 

### 📦 Descarga de Modelos

Puedes descargar los 3 archivos necesarios desde MEGA:

**🔗 [Descargar modelos desde MEGA](https://mega.nz/file/iwwkQDQI#veGcq2FKE9nk35A8hZfYX3j9o60GHfJYkyGAdkjnOr4)**

Los archivos incluidos son:
- `weather_model.pkl` - Modelo Random Forest entrenado
- `weather_scaler.pkl` - StandardScaler para normalización de features
- `weather_config.pkl` - Configuración y metadatos del modelo

**Instrucciones de uso:**
1. Descarga el archivo desde el link de MEGA
2. Extrae los archivos `.pkl` en la carpeta `models/` del proyecto
3. Ejecuta la API con `uvicorn app.main:app --reload`

---

## 🏗️ Arquitectura del Sistema

### Backend - API FastAPI
- **Framework**: FastAPI con soporte CORS
- **Modelo ML**: Random Forest MultiOutput (scikit-learn)
- **Normalización**: StandardScaler
- **Endpoints REST**: Predicciones por coordenadas y fecha/hora
- **Documentación**: Swagger UI automática en `/docs`

### Aplicación de Pruebas - Streamlit
- **Mapa interactivo**: Selección visual de ubicaciones en Ecuador
- **Predicciones en tiempo real**: Visualización de las 6 variables climáticas
- **Gráficos**: Charts interactivos con Plotly
- **Caching**: Optimización de carga del modelo

### Estructura del Proyecto
