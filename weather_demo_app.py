"""
Aplicación Demo Streamlit para Predicción Climática
Prueba el modelo de clasificación Soleado vs Nublado
"""
import streamlit as st
from datetime import datetime, date
import pandas as pd
from weather_demo_api import predictor

# Configuración de página
st.set_page_config(
    page_title="Demo Predicción Clima Ecuador",
    page_icon="🌤️",
    layout="wide"
)

# CSS personalizado
st.markdown("""
<style>
    .big-font {
        font-size: 60px !important;
        text-align: center;
        margin: 20px 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .sunny {
        background: linear-gradient(135deg, #FDB99B 0%, #FCF6B1 100%);
        color: #333;
    }
    .cloudy {
        background: linear-gradient(135deg, #A8C0D4 0%, #DDE8F0 100%);
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🌤️ Predicción Climática - Ecuador")
    st.markdown("### Sistema de Predicción: Soleado ☀️ vs Nublado ☁️")
    
    # Mostrar modo del predictor
    if predictor.use_ml_model:
        st.success("✅ Modelo ML (Random Forest) cargado - Accuracy: ~88%")
    else:
        st.warning("⚠️  Modo Demo: Usando predictor basado en reglas (subir weather_model.pkl para usar ML)")
    
    # Separador
    st.markdown("---")
    
    # Layout en dos columnas
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📍 Ubicación")
        
        # Opciones de ciudades predefinidas
        ciudades = {
            "Quito": {"lat": -0.1807, "lon": -78.4678},
            "Guayaquil": {"lat": -2.1710, "lon": -79.9224},
            "Cuenca": {"lat": -2.9001, "lon": -79.0059},
            "Manta": {"lat": -0.9677, "lon": -80.7089},
            "Loja": {"lat": -3.9932, "lon": -79.2057},
            "Personalizado": {"lat": 0.0, "lon": -78.0}
        }
        
        ciudad_seleccionada = st.selectbox(
            "Selecciona una ciudad o usa coordenadas personalizadas",
            list(ciudades.keys())
        )
        
        if ciudad_seleccionada == "Personalizado":
            latitud = st.number_input(
                "Latitud",
                min_value=-5.0,
                max_value=5.0,
                value=0.0,
                step=0.1,
                help="Rango válido para Ecuador: -5.0 a 5.0"
            )
            longitud = st.number_input(
                "Longitud",
                min_value=-81.0,
                max_value=-75.0,
                value=-78.0,
                step=0.1,
                help="Rango válido para Ecuador: -81.0 a -75.0"
            )
        else:
            latitud = ciudades[ciudad_seleccionada]["lat"]
            longitud = ciudades[ciudad_seleccionada]["lon"]
            st.info(f"📍 {ciudad_seleccionada}: Lat {latitud:.4f}, Lon {longitud:.4f}")
    
    with col2:
        st.subheader("📅 Fecha y Hora")
        
        fecha_input = st.date_input(
            "Fecha",
            value=date.today(),
            help="Selecciona la fecha para la predicción"
        )
        
        hora_input = st.slider(
            "Hora del día",
            min_value=0,
            max_value=23,
            value=12,
            help="0 = medianoche, 12 = mediodía"
        )
        
        # Mostrar hora en formato legible
        hora_formato = datetime.strptime(str(hora_input), "%H").strftime("%I:00 %p")
        st.info(f"🕐 Hora seleccionada: {hora_formato}")
    
    # Botón de predicción
    st.markdown("---")
    
    if st.button("🔮 Predecir Clima", use_container_width=True, type="primary"):
        
        with st.spinner("Analizando condiciones meteorológicas..."):
            try:
                # Hacer predicción
                resultado = predictor.predict(
                    lat=latitud,
                    lon=longitud,
                    fecha=fecha_input.strftime('%Y-%m-%d'),
                    hora=hora_input
                )
                
                # Mostrar resultado
                st.markdown("---")
                st.markdown("## 🎯 Resultado de la Predicción")
                
                # Predicción principal
                prediccion = resultado['prediccion']
                
                if prediccion['es_soleado']:
                    emoji = "☀️"
                    clase = "sunny"
                    mensaje = "¡Día Soleado!"
                else:
                    emoji = "☁️"
                    clase = "cloudy"
                    mensaje = "Día Nublado"
                
                st.markdown(f'<div class="metric-card {clase}"><div class="big-font">{emoji}</div><h2 style="text-align:center;">{mensaje}</h2></div>', unsafe_allow_html=True)
                
                # Métricas
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Confianza",
                        f"{prediccion['confianza_pct']}%",
                        help="Nivel de confianza del modelo"
                    )
                
                with col2:
                    st.metric(
                        "Prob. Soleado",
                        f"{prediccion['probabilidad_soleado_pct']}%",
                        help="Probabilidad de condiciones soleadas"
                    )
                
                with col3:
                    st.metric(
                        "Prob. Nublado",
                        f"{prediccion['probabilidad_nublado_pct']}%",
                        help="Probabilidad de condiciones nubladas"
                    )
                
                # Datos meteorológicos
                st.markdown("### 📊 Datos Meteorológicos (Estimados)")
                
                datos_met = resultado['datos_meteorologicos']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "🌡️ Temperatura",
                        f"{datos_met['temperatura_c']}°C"
                    )
                
                with col2:
                    st.metric(
                        "💧 Humedad",
                        f"{datos_met['humedad_pct']}%"
                    )
                
                with col3:
                    st.metric(
                        "💨 Viento",
                        f"{datos_met['viento_ms']} m/s"
                    )
                
                # Detalles técnicos (expandible)
                with st.expander("🔧 Detalles Técnicos"):
                    st.json(resultado)
                
            except Exception as e:
                st.error(f"❌ Error al hacer la predicción: {str(e)}")
    
    # Información del modelo
    st.markdown("---")
    st.markdown("### ℹ️ Información del Sistema")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Modelo:** Random Forest Classifier
        
        **Accuracy:** ~88%
        
        **Features utilizadas:**
        - Temperatura
        - Humedad
        - Velocidad del viento
        - Hora del día
        - Mes y día del año
        """)
    
    with col2:
        st.info("""
        **Región:** Ecuador
        
        **Latitud:** -5° a 5°
        
        **Longitud:** -81° a -75°
        
        **Datos:** 2M+ registros (2020-2024)
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🌤️ Sistema de Predicción Climática para Ecuador</p>
        <p>Desarrollado con datos satelitales de NASA POWER</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
