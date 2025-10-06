#  It Will Rain on My Parade?  
### Proyecto presentado en el *NASA Space Apps Challenge Hackathon 2025*

Desarrollado por el *Team: Weather_Forecast*

 *Integrantes:*
- Daheny Lopez  
- Jhony Peñaherrera  
- Silvia García  
- Lander Lliguicota  
- Daniel Rivas  
- Mishell Guano

---

##  Descripción del Proyecto

Sistema de predicción climática para Ecuador que utiliza un modelo **Random Forest MultiOutput** entrenado con datos históricos de NASA POWER. El sistema predice 6 variables climáticas simultáneamente para cualquier ubicación dentro del territorio ecuatoriano.

###  Variables Predichas
- **T2M**: Temperatura a 2 metros (°C)
- **RH2M**: Humedad relativa a 2 metros (%)
- **WS10M**: Velocidad del viento a 10 metros (m/s)
- **CLOUD_AMT**: Cantidad de nubosidad (%)
- **ALLSKY_SFC_SW_DWN**: Radiación solar descendente (W/m²)
- **PoP**: Probabilidad de precipitación (%)


---

##  Pruebas del Modelo - Aplicación Streamlit

Para validar el funcionamiento del modelo Random Forest, desarrollamos una **aplicación demo en Streamlit** que permite:

- Seleccionar coordenadas dentro de Ecuador mediante un mapa interactivo
- Elegir fecha y hora específicas para la predicción
- Visualizar las 6 variables climáticas predichas en tiempo real
- Verificar la precisión del modelo con datos reales

### Rendimiento del Modelo
El modelo Random Forest MultiOutput fue entrenado con un dataset de **2+ millones de registros** históricos y alcanzó los siguientes scores R²:

- Temperatura (T2M): **0.95**
- Humedad (RH2M): **0.89**
- Velocidad del viento (WS10M): **0.72**
- Nubosidad (CLOUD_AMT): **0.68**
- Radiación solar (ALLSKY_SFC_SW_DWN): **0.91**
- Probabilidad de precipitación (PoP): **0.45**

###  Demo en Streamlit
La aplicación Streamlit (`weather_demo_app.py`) sirvió como entorno de pruebas para validar:
- Carga correcta de los modelos entrenados
- Procesamiento de coordenadas geográficas
- Escalado de features con StandardScaler
- Predicciones en tiempo real
- Visualización interactiva de resultados

---

### Estructura de los modelos desarrollados:

| Nombre del archivo | Descripción | Último commit |
|--------------------|-------------|----------------|
| `random_forest.py` | Script para el entrenamiento y evaluación del modelo **Random Forest** utilizando las variables meteorológicas procesadas. | `Add files via upload` |
| `train_weather_lstm.py` | Script para el entrenamiento del modelo **LSTM (RNN)** enfocado en la predicción de condiciones climáticas a partir de series temporales. | `Add files via upload` |

---

###  Descripción general

Esta carpeta contiene los scripts principales utilizados para **el modelamiento y entrenamiento de predictores climáticos** basados en datos de la NASA.  
Los modelos desarrollados incluyen:

- **Random Forest:** para el análisis de patrones no lineales y predicción a corto plazo.  
- **LSTM (Long Short-Term Memory):** red neuronal recurrente para la predicción secuencial del clima a partir de variables meteorológicas históricas.

---

### ⚙️ Modificación del dataset

Durante el proceso de desarrollo, **la base de datos fue modificada y extendida**.  
Se aplicaron diversas técnicas de generación y enriquecimiento de datos para **incluir exitosamente la variable de precipitación (`pop`)**, lo cual permitió mejorar la precisión del modelo.

---

### Acceso al dataset

Debido al gran tamaño del conjunto de datos, el archivo CSV original se encuentra disponible en Google Drive:  
📁 [Descargar dataset (.csv)](https://drive.google.com/file/d/1r2yINzDHarD1uLNHs_9cizrZYXw3FqB_/view?usp=sharing)
##  Archivos del Modelo (.pkl)

---

Debido al tamaño considerable de los archivos del modelo entrenado (>100MB), **no fue posible subirlos directamente a GitHub**. 

###  Descarga del Modelo

Puedes descargar el  archivo necesario desde MEGA:

**🔗 [Descargar modelos desde MEGA](https://mega.nz/file/iwwkQDQI#veGcq2FKE9nk35A8hZfYX3j9o60GHfJYkyGAdkjnOr4)**

**Instrucciones de uso:**
1. Descarga el archivo desde el link de MEGA
2. Extrae los archivos `.pkl` en la carpeta `models/` del proyecto
3. Ejecuta la API con `uvicorn app.main:app --reload`

---

## Anexos:
<img width="1920" height="1044" alt="image" src="https://github.com/user-attachments/assets/4f4f07d2-87e7-485f-be91-7bc08048af44" />
<img width="1920" height="1056" alt="image" src="https://github.com/user-attachments/assets/791c3682-c5a8-4d01-b57d-c325ee3084a7" />

## 🛰️ Créditos y reconocimiento

**Proyecto:** *It Will Rain on My Parade?*  
**Equipo:** *Weather_Forecast* — NASA Space Apps Challenge Hackathon 2025  
**Categoría:** Ciencia de datos, predicción meteorológica y visualización geoespacial.


