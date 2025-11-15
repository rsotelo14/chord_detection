# Interfaz Web - Detector de Acordes

Interfaz web para detectar acordes de canciones en tiempo real.

## Instalación

1. Asegúrate de tener el entorno virtual activado del proyecto principal:
```bash
cd ..
source env/bin/activate  # En Windows: env\Scripts\activate
```

2. Instala las dependencias adicionales para el servidor web:
```bash
pip install -r web/requirements.txt
```

## Uso

1. Desde el directorio `web/`, ejecuta:
```bash
python app.py
```

2. Abre tu navegador en: `http://localhost:5000`

3. Sube un archivo de audio (MP3, WAV, OGG, M4A, FLAC)

4. Espera a que el modelo procese la canción

5. ¡Reproduce y toca junto con los acordes detectados!

## Características

- ✨ Interfaz moderna y responsive
- 🎵 Reproductor de audio integrado
- 🎸 Acordes sincronizados en tiempo real
- 📊 Línea de tiempo interactiva
- 🖱️ Drag & drop para subir archivos
- 🔄 Procesamiento automático con el modelo MLP

## Limitaciones

- Tamaño máximo de archivo: 50MB
- Formatos soportados: MP3, WAV, OGG, M4A, FLAC
- Timeout de procesamiento: 5 minutos
































