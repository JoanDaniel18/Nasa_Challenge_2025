"""
Descarga el modelo desde MEGA, Google Drive u otro servicio
Ejecuta esto EN REPLIT después de subir tu modelo
"""
import requests
import os

def download_from_mega(url, filename="weather_model.pkl"):
    """
    Descarga el modelo desde MEGA o cualquier URL directa
    
    Args:
        url: Link de descarga de MEGA (con la key incluida)
        filename: Nombre del archivo destino
    """
    print(f"📥 Descargando modelo...")
    print(f"   URL: {url[:50]}...")
    
    try:
        # Descargar con streaming para archivos grandes
        response = requests.get(url, stream=True, allow_redirects=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as f:
            downloaded = 0
            chunk_size = 8192
            
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Mostrar progreso
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        mb_downloaded = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        print(f"   Progreso: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='\r')
        
        print()  # Nueva línea después del progreso
        
        # Verificar que se descargó
        if os.path.exists(filename):
            size_mb = os.path.getsize(filename) / (1024 * 1024)
            print(f"✅ Modelo descargado exitosamente: {filename}")
            print(f"📦 Tamaño: {size_mb:.2f} MB")
            return filename
        else:
            print("❌ Error: El archivo no se descargó correctamente")
            return None
            
    except Exception as e:
        print(f"❌ Error al descargar: {e}")
        print("\n💡 Verifica que:")
        print("   1. El link sea de descarga directa (no de visualización)")
        print("   2. El archivo sea público/compartido")
        print("   3. Tienes conexión a internet")
        return None

if __name__ == "__main__":
    print("="*60)
    print("📥 Descargador de Modelo ML")
    print("="*60)
    
    # REEMPLAZA CON TU LINK DE MEGA, GOOGLE DRIVE, ETC.
    # Para MEGA: Usa el link completo que te dan (incluye la key)
    # Para Google Drive: Usa formato https://drive.google.com/uc?export=download&id=FILE_ID
    
    MODEL_URL = "https://mega.nz/file/iwwkQDQI#veGcq2FKE9nk35A8hZfYX3j9o60GHfJYkyGAdkjnOr4"
    
    if MODEL_URL == "TU_LINK_AQUI":
        print("⚠️  Por favor edita este archivo y reemplaza MODEL_URL con tu link")
        print("\nPasos:")
        print("1. Sube weather_model.pkl a MEGA o Google Drive")
        print("2. Obtén el link de descarga pública")
        print("3. Edita download_from_huggingface.py y pega el link")
        print("4. Ejecuta: python3 download_from_huggingface.py")
    else:
        download_from_mega(MODEL_URL)
        print("\n✨ ¡Listo! Reinicia la app para usar el modelo ML")
