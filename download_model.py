"""
Script para descargar el modelo desde un servicio externo
Ejecuta esto EN REPLIT después de subir tu modelo a algún lugar
"""
import requests
import os

def download_from_url(url, filename='weather_model.pkl'):
    """
    Descarga el modelo desde una URL
    
    Opciones para hospedar tu modelo:
    1. Google Drive (genera link público)
    2. Dropbox (genera link público)
    3. GitHub Release (para archivos < 100MB)
    4. WeTransfer u otro servicio de transferencia
    """
    print(f"Descargando modelo desde: {url}")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                # Mostrar progreso
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"Progreso: {percent:.1f}%", end='\r')
    
    print(f"\n✅ Modelo descargado: {filename}")
    print(f"Tamaño: {os.path.getsize(filename) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    # REEMPLAZA CON TU URL
    model_url = "https://tu-url-aqui.com/weather_model.pkl"
    
    # Ejemplo para Google Drive:
    # model_url = "https://drive.google.com/uc?export=download&id=TU_FILE_ID"
    
    download_from_url(model_url)
