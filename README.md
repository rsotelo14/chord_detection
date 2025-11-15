# 🎸 Detector de Acordes - Proyecto ML

Sistema de detección automática de acordes en música usando técnicas de Machine Learning. Este proyecto implementa dos modelos principales: un baseline MLP que trabaja con segmentos de audio agrupados por beats, y un modelo basado en frames que procesa el audio frame por frame con mayor resolución temporal.

## 📋 Tabla de Contenidos

- [Requerimientos](#requerimientos)
- [Instalación](#instalación)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Modelo Baseline MLP](#modelo-baseline-mlp)
  - [Crear Dataset](#crear-dataset-baseline-mlp)
  - [Entrenar Modelo](#entrenar-modelo-baseline-mlp)
  - [Inferencia](#inferencia-baseline-mlp)
  - [Evaluación](#evaluación-baseline-mlp)
- [Modelo de Frames](#modelo-de-frames)
  - [Crear Dataset](#crear-dataset-frames)
  - [Entrenar Modelo](#entrenar-modelo-frames)
  - [Inferencia](#inferencia-frames)
  - [Evaluación](#evaluación-frames)
- [Interfaz Web](#interfaz-web)
- [Estructura de Datos](#estructura-de-datos)

## 🔧 Requerimientos

### Dependencias Python

El proyecto requiere Python 3.9+ y las siguientes librerías principales:

- **TensorFlow/Keras**: Para los modelos de redes neuronales
- **librosa**: Para procesamiento de audio y extracción de features
- **scikit-learn**: Para preprocesamiento y evaluación
- **pandas/numpy**: Para manipulación de datos
- **matplotlib**: Para visualizaciones
- **Flask**: Para la interfaz web

### Instalación de Dependencias

```bash
# Crear entorno virtual (recomendado)
python -m venv env

# Activar entorno virtual
# En Mac/Linux:
source env/bin/activate
# En Windows:
env\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### Datos Requeridos

El proyecto espera tener los datos de The Beatles Annotations en la siguiente estructura:

```
The Beatles Annotations/
├── audio/
│   └── The Beatles/
│       └── [álbumes y canciones .mp3]
├── chordlab/
│   └── The Beatles/
│       └── [archivos .lab con anotaciones de acordes]
└── ...
```

## 📁 Estructura del Proyecto

```
chord_detection/
├── The Beatles Annotations/     # Dataset de audio y anotaciones
├── analysis_out/                # Resultados del modelo baseline MLP
├── analysis_out_frames/         # Resultados del modelo de frames
├── outputs/                     # Archivos .lab generados por inferencia
├── baseline_mlp.py              # Script de entrenamiento baseline MLP
├── build_dataset.py             # Generación de dataset para baseline MLP
├── build_frames_dataset.py      # Generación de dataset para modelo frames
├── train_dnn_frames.py          # Script de entrenamiento modelo frames
├── inference_baseline_mlp.py    # Script de inferencia baseline MLP
├── inference_frames.py          # Script de inferencia modelo frames
├── evaluate_wcsr.py            # Script de evaluación WCSR
├── train_test_split.json        # División train/test por canción
├── dataset_chords.csv           # Dataset generado para baseline MLP
├── frames_dataset.csv           # Dataset generado para modelo frames
├── frames_dataset.npz           # Dataset binario para modelo frames
├── pca.joblib                   # Modelo PCA para frames
├── scaler.joblib                # Scaler para frames
├── web/                         # Interfaz web Flask
│   ├── app.py
│   ├── templates/
│   └── static/
└── requirements.txt
```

## 🎯 Modelo Baseline MLP

El modelo baseline MLP trabaja con segmentos de audio agrupados por beats. Cada segmento representa varios beats (por defecto 4) y se caracteriza por un vector de cromas agregado.

### Crear Dataset (Baseline MLP)

El script `build_dataset.py` procesa los archivos de audio y genera un CSV con features de cromas agregadas por segmento de acorde.

```bash
python build_dataset.py
```

**Salida:**
- `dataset_chords.csv`: Dataset con columnas:
  - `album_track`: Identificador de la canción
  - `t_start`, `t_end`: Tiempos de inicio y fin del segmento
  - `label`: Etiqueta del acorde (formato `Root:maj` o `Root:min`)
  - `chroma_C`, `chroma_Db`, ..., `chroma_B`: Valores de cromas (12 dimensiones)

**Características:**
- Usa chroma CQT estándar de 12 bins (una por nota)
- Agrega cromas por mediana dentro de cada intervalo de acorde
- Filtra segmentos con label "N" (sin acorde)
- Normaliza cromas por columna

### Entrenar Modelo (Baseline MLP)

```bash
python baseline_mlp.py
```

**Proceso:**
1. Carga el dataset desde `dataset_chords.csv`
2. Divide train/test usando `train_test_split.json` (por canción)
3. Divide train en train/validation (80/20 por canción)
4. Estandariza features con `StandardScaler`
5. Entrena dos arquitecturas:
   - **MLP Original**: 2 capas densas (128 unidades cada una)
   - **MLP Bottleneck**: Arquitectura con compresión-expansión (128→64→32→64→128)
6. Selecciona el mejor modelo según accuracy en test
7. Genera métricas y visualizaciones

**Salidas en `analysis_out/`:**
- `baseline_mlp_model.h5`: Modelo entrenado (mejor arquitectura)
- `mlp_label_mapping.txt`: Mapeo de índices a nombres de clases
- `mlp_scaler_stats.npz`: Estadísticas del scaler (media y desviación estándar)
- `baseline_mlp_confusion.png`: Matriz de confusión
- `baseline_mlp_loss.png`: Curvas de pérdida
- `baseline_mlp_predictions.csv`: Predicciones en test
- `best_model_report_*.txt`: Reporte de clasificación del mejor modelo

**Parámetros configurables en el script:**
- `BATCH = 64`: Tamaño de batch
- `EPOCHS = 100`: Número máximo de épocas
- `DROPOUT = 0.40`: Tasa de dropout
- `TEST_SIZE = 0.25`: Proporción de test

### Inferencia (Baseline MLP)

```bash
python inference_baseline_mlp.py <ruta_al_audio> [opciones]
```

**Opciones principales:**
- `--model`: Ruta al modelo `.h5` (default: `analysis_out/baseline_mlp_model.h5`)
- `--labels`: Ruta al mapeo de labels (default: `analysis_out/mlp_label_mapping.txt`)
- `--scaler`: Ruta al scaler stats (default: `analysis_out/mlp_scaler_stats.npz`)
- `--beats-per-segment`: Número de beats por segmento (default: 4)
- `--use-hmm`: Usar HMM para suavizar predicciones
- `--hmm`: Ruta al modelo HMM (default: `analysis_out/hmm_model.npz`)
- `--transition-weight`: Peso de transiciones HMM (default: 0.3)
- `--out`: Prefijo de salida (default: `outputs/<nombre_audio>`)

**Ejemplo:**
```bash
python inference_baseline_mlp.py test_audios/cancion.mp3 --beats-per-segment 4 --use-hmm
```

**Salida:**
- Archivo `.lab` en `outputs/` con predicciones en formato:
  ```
  t_start    t_end      label_pred
  0.000000   2.500000   C:maj
  2.500000   5.000000   G:maj
  ...
  ```

### Evaluación (Baseline MLP)

```bash
python evaluate_wcsr.py baseline_mlp <split> [opciones]
```

**Ejemplo:**
```bash
# Evaluar en test split
python evaluate_wcsr.py baseline_mlp test

# Evaluar en train split con HMM
python evaluate_wcsr.py baseline_mlp train --use-hmm-baseline --beats-per-segment 4

# Limitar cantidad de canciones para prueba rápida
python evaluate_wcsr.py baseline_mlp test --max-songs 10
```

**Opciones:**
- `split`: `train` o `test`
- `--beats-per-segment`: Número de beats por segmento (default: 4)
- `--use-hmm-baseline`: Usar HMM para suavizar
- `--transition-weight`: Peso de transiciones HMM (default: 0.3)
- `--max-songs`: Límite de canciones a evaluar

**Salidas:**
- CSV con resultados por canción en `analysis_out/wcsr_*_beats_results.csv`
- Archivos `.lab` predichos en `outputs/` o `outputs_test/`
- Métricas: WCSR global, promedio, mediana, desviación estándar

## 🎼 Modelo de Frames

El modelo de frames procesa el audio frame por frame con mayor resolución temporal. Usa CQT (Constant-Q Transform) con splicing de contexto temporal.

### Crear Dataset (Frames)

```bash
python build_frames_dataset.py
```

**Proceso:**
1. Procesa cada archivo de audio frame por frame
2. Calcula CQT log-magnitude (180 bins, 5 octavas)
3. Aplica PCA (retención 98% de varianza) solo en train
4. Estandariza con `StandardScaler` solo en train
5. Aplica splicing de contexto (t-1, t, t+1) → 3 frames concatenados
6. Genera labels por frame desde anotaciones `.lab`

**Salidas:**
- `frames_dataset.csv`: Metadatos (album_track, tiempo, label)
- `frames_dataset.npz`: Dataset binario con:
  - `X`: Features (N, F') donde F' = (2*ctx+1) * PCA_dims
  - `y`: Labels (N,)
  - `groups`: Identificadores de canción
  - `times`: Tiempos de cada frame
- `pca.joblib`: Modelo PCA entrenado
- `scaler.joblib`: Scaler entrenado
- `analysis_out_frames/class_balance_counts_frames.csv`: Distribución de clases

**Características:**
- Sample rate: 11025 Hz
- Hop length: 512 (~46.4 ms por frame)
- CQT: 180 bins (36 bins/octava × 5 octavas)
- Contexto: ±1 frame (SPLICE_CTX=1)
- Filtra frames con label "N"

### Entrenar Modelo (Frames)

```bash
python train_dnn_frames.py
```

**Proceso:**
1. Carga dataset desde `frames_dataset.npz`
2. Divide train/test por canción (75/25)
3. Divide train en train/validation (80/20 por canción)
4. Entrena dos arquitecturas:
   - **MLP Común**: 2 capas densas (1024 unidades cada una)
   - **MLP Bottleneck**: Arquitectura con compresión-expansión (1024→512→256→512→1024)
5. Usa class weights balanceados
6. Selecciona mejor modelo según accuracy

**Salidas en `analysis_out_frames/`:**
- `dnn_common.h5`: Modelo MLP común
- `dnn_bottleneck.h5`: Modelo MLP bottleneck
- `label_mapping.txt`: Mapeo de índices a nombres de clases
- `class_weights.txt`: Pesos de clases balanceados
- `train_val_loss.png`: Curvas de pérdida comparativas
- `report_common.txt`: Reporte de clasificación (común)
- `report_bottleneck.txt`: Reporte de clasificación (bottleneck)

**Parámetros configurables:**
- `BATCH = 128`: Tamaño de batch
- `EPOCHS = 50`: Número máximo de épocas
- Dropout: 0.3 en todas las capas

### Inferencia (Frames)

```bash
python inference_frames.py <ruta_al_audio> [opciones]
```

**Opciones:**
- `--smooth`: Usar HMM para suavizar predicciones (default: activado)
- `--beat-sync`: Alinear predicciones a beats y usar majority voting
- `--beat-group`: Cantidad de beats por grupo para voting (default: 2)

**Ejemplo:**
```bash
# Inferencia básica con HMM
python inference_frames.py test_audios/cancion.mp3 --smooth

# Inferencia con beat-sync
python inference_frames.py test_audios/cancion.mp3 --smooth --beat-sync --beat-group 4

# Sin HMM
python inference_frames.py test_audios/cancion.mp3
```

**Rutas por defecto (configurables en el script):**
- `PCA = Path("pca.joblib")`
- `SCAL = Path("scaler.joblib")`
- `MODEL = Path("analysis_out_frames/dnn_bottleneck.h5")`
- `MAP = Path("analysis_out_frames/label_mapping.txt")`

**Salida:**
- Archivo `.lab` en `outputs/` con predicciones frame por frame fusionadas

### Evaluación (Frames)

```bash
python evaluate_wcsr.py frames <split> [opciones]
```

**Ejemplo:**
```bash
# Evaluar en test split
python evaluate_wcsr.py frames test

# Evaluar sin HMM
python evaluate_wcsr.py frames test --no-hmm-frames

# Evaluar con beat-sync
python evaluate_wcsr.py frames_beatsync test --beat-group 4
```

**Opciones:**
- `split`: `train` o `test`
- `--use-hmm-frames`: Usar HMM (default: True)
- `--no-hmm-frames`: Desactivar HMM
- `--beat-group`: Para `frames_beatsync`, cantidad de beats por grupo

**Salidas:**
- CSV con resultados en `analysis_out_frames/wcsr_*_results.csv`
- Archivos `.lab` predichos en `outputs/` o `outputs_test/`
- Métricas: WCSR promedio, mediana, desviación estándar

## 🌐 Interfaz Web

La interfaz web permite subir archivos de audio y visualizar los acordes detectados en tiempo real.

### Instalación de Dependencias Web

```bash
cd web
pip install -r requirements.txt
```

### Ejecutar Servidor Web

**Opción 1: Scripts de inicio (recomendado)**

En Mac/Linux:
```bash
cd web
./start.sh
```

En Windows:
```bash
cd web
start.bat
```

**Opción 2: Manual**
```bash
cd web
python app.py
```

El servidor se iniciará en `http://localhost:5000`

### Uso de la Interfaz Web

1. **Abrir navegador**: Navega a `http://localhost:5000`
2. **Subir audio**: Arrastra y suelta un archivo o haz clic para seleccionar
   - Formatos soportados: MP3, WAV, OGG, M4A, FLAC
   - Tamaño máximo: 50MB
3. **Procesamiento**: El modelo procesará el audio automáticamente
4. **Visualización**: Los acordes se mostrarán sincronizados con el reproductor

**Características:**
- Reproductor de audio integrado
- Visualización de acordes en tiempo real
- Línea de tiempo interactiva
- Click en acorde para saltar a ese momento

**Nota**: La interfaz web usa el modelo de frames (`inference_frames.py`) con HMM y beat-sync activados por defecto.

## 📊 Estructura de Datos

### Formato de Archivos .lab

Los archivos `.lab` contienen anotaciones de acordes en formato:

```
t_start    t_end      label
0.000000   2.500000   C:maj
2.500000   5.000000   G:maj
5.000000   7.500000   A:min
...
```

- **t_start, t_end**: Tiempos en segundos (formato float)
- **label**: Etiqueta del acorde en formato `Root:maj` o `Root:min`
  - Raíces: C, Db, D, Eb, E, F, Gb, G, Ab, A, Bb, B
  - Calidades: `maj` (mayor) o `min` (menor)

### Formato train_test_split.json

```json
{
  "train_songs": [
    "album1/track1",
    "album1/track2",
    ...
  ],
  "test_songs": [
    "album2/track1",
    "album2/track2",
    ...
  ]
}
```

Este archivo define la división train/test por canción para evitar data leakage.

## 🔍 Métricas de Evaluación

### WCSR (Weighted Chord Symbol Recall)

El WCSR mide la proporción de tiempo donde las etiquetas predichas coinciden con las de referencia, excluyendo segmentos marcados como "N" (sin acorde).

- **WCSR Global**: Ponderado por duración total
- **WCSR Promedio**: Promedio aritmético por canción
- **WCSR Mediana**: Mediana de WCSR por canción

## 📝 Notas Adicionales

- Los modelos se entrenan con **class weights balanceados** para manejar desbalance de clases
- Se usa **early stopping** y **reducción de learning rate** durante el entrenamiento
- El **HMM** se puede usar para suavizar predicciones y mejorar coherencia temporal
- El modelo de frames tiene mayor resolución temporal pero requiere más recursos computacionales
- El modelo baseline MLP es más rápido pero con menor resolución temporal

## 🐛 Solución de Problemas

### Error: "No se encontró el modelo"
Asegúrate de haber entrenado el modelo correspondiente antes de ejecutar inferencia o evaluación.

### Error: "No module named 'tensorflow'"
Instala las dependencias: `pip install -r requirements.txt`

### Error: "FileNotFoundError: dataset_chords.csv"
Ejecuta primero `build_dataset.py` para generar el dataset.

### El puerto 5000 está ocupado (web)
Edita `web/app.py` y cambia el puerto:
```python
app.run(debug=True, port=8000)  # Usar puerto 8000
```

## 📚 Referencias

- [librosa](https://librosa.org/) - Procesamiento de audio
- [TensorFlow/Keras](https://www.tensorflow.org/) - Deep Learning
- [The Beatles Annotations Dataset](https://github.com/tmc323/Chord-Annotations)

---

**Autor**: Proyecto de Machine Learning - Detección de Acordes  
**Fecha**: 2025

