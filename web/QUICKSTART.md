# 🎸 Inicio Rápido - Detector de Acordes Web

## Instalación en 2 pasos

### 1. Instalar dependencias de Flask
```bash
cd web
pip install -r requirements.txt
```

### 2. Iniciar el servidor

**En Mac/Linux:**
```bash
./start.sh
```

**En Windows:**
```bash
start.bat
```

**O manualmente:**
```bash
python app.py
```

## Uso

1. Abre tu navegador en: **http://localhost:5000**

2. **Sube una canción:**
   - Arrastra y suelta un archivo de audio
   - O haz clic para seleccionar un archivo
   - Formatos: MP3, WAV, OGG, M4A, FLAC (máx. 50MB)

3. **Espera el procesamiento:**
   - El modelo analizará la canción (puede tomar algunos segundos)

4. **¡Toca junto con los acordes!**
   - Reproduce la canción
   - Los acordes se mostrarán en tiempo real
   - Haz clic en cualquier acorde para saltar a ese momento

## Solución de problemas

### Error: "No se encontró el modelo"
Asegúrate de haber entrenado el modelo primero:
```bash
cd ..
python baseline_mlp.py
```

### Error: "No module named flask"
Instala las dependencias:
```bash
pip install -r requirements.txt
```

### El puerto 5000 está ocupado
Edita `app.py` y cambia el puerto:
```python
app.run(debug=True, port=8000)  # Usar puerto 8000 en lugar de 5000
```

## Características

- ✨ Interfaz moderna y responsive
- 🎵 Reproductor de audio integrado
- 🎸 Acordes sincronizados en tiempo real
- 📊 Línea de tiempo interactiva
- 🖱️ Drag & drop para subir archivos
- 🔄 Procesamiento automático con el modelo MLP

## Tecnologías

- **Backend:** Flask (Python)
- **Frontend:** HTML5, CSS3, JavaScript vanilla
- **Audio:** Web Audio API
- **ML:** TensorFlow/Keras (modelo MLP pre-entrenado)
































