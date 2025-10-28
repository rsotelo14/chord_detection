# 📊 Log de Experimentos - Chord Detection

## 🎯 Resumen de Resultados

| Experimento | Accuracy | Macro F1 | Balanced Acc | WCSR (Independiente) | Fecha | Notas |
|-------------|----------|----------|--------------|---------------------|-------|-------|
| **MLP Original (menos datos: ~2000)** | 73.16% | 61.85% | 66.59% | 50.46% | - | Baseline inicial |
| **MLP Mejorado (más datos ~5000)** | 69.70% | 51.90% | 63.50% | 59.97% | - | +Más canciones Beatles |
| **MLP + HMM** | 69.70% | 51.90% | 63.50% | **61.81%** | - | +HMM post-processing |
|
| **MLP + HMM (mas datos: ~11000)** | 69.70% | 51.90% | 63.50% | **61.81%** | - | +HMM post-processing |
| **DNN Frames (CQT + PCA + HMM)** | - | - | - | **78.43%** | 2024 | +Frame-wise + CQT + PCA + Splicing |
|
## 📈 Progreso por Configuración

### 🏆 **Mejor Configuración Actual**
- **Modelo**: DNN Frames (CQT + PCA + HMM)
- **WCSR Independiente**: **78.43%**
- **Dataset**: Beatles (frame-wise, ~500K frames)
- **Post-processing**: HMM con stay_prob=0.995
- **Features**: CQT → PCA → Splicing (±1 frame)

### 📊 **Detalles por Experimento**

#### 1. MLP Original (Baseline)
- **Dataset**: Dataset original
- **Arquitectura**: 12 → 128 → 64 → 24
- **WCSR Independiente**: 50.46%
- **Canciones test**: Come Together, Misery, Please Please Me, Love Me Do

#### 2. MLP Mejorado (+Más Datos)
- **Dataset**: Ampliado con más canciones Beatles
- **Arquitectura**: 12 → 128 → 64 → 24
- **WCSR Independiente**: 59.97% (+9.51%)
- **Mejora**: Agregar más canciones de entrenamiento

#### 3. MLP + HMM
- **Dataset**: Ampliado con más canciones Beatles
- **Arquitectura**: 12 → 128 → 64 → 24 + HMM
- **WCSR Independiente**: 61.81% (+11.35% vs baseline)
- **HMM**: transition_weight=0.3
- **Canciones test**: For You Blue, Misery, Please Please Me, Love Me Do

#### 4. MLP Bottleneck
- **Dataset**: Ampliado con más canciones Beatles
- **Arquitectura**: 12 → 128 → 64 → 32 → 64 → 128 → 24
- **Resultado**: ❌ Empeoró en todas las métricas
- **Conclusión**: Arquitectura más simple es mejor para este dataset

#### 5. DNN Frames (CQT + PCA + HMM)
- **Dataset**: Beatles (frame-wise, ~500K frames)
- **Arquitectura**: MLP 1024→1024→24 (bottleneck: 1024→512→256→512→1024→24)
- **Features**: CQT (180 bins) → PCA (~120 dims) → Splicing (±1 frame)
- **WCSR Independiente**: **78.43%** (+16.62% vs MLP+HMM anterior)
- **Mejora**: Frame-wise en vez de beats-per-segment + features CQT + context splicing
- **Canciones test**: For You Blue (57.5%), Misery (80.4%), Please Please Me (86.5%), Love Me Do (92.2%)

## 🎵 **WCSR por Canción (Conjunto Independiente)**

### Configuración Anterior (MLP + HMM)
| Canción | WCSR | Duración | Segmentos GT | Segmentos Pred |
|---------|------|----------|--------------|----------------|
| **Please Please Me** | 66.0% | 119.0s | 77 | 38 |
| **Misery** | 63.9% | 105.7s | 44 | 42 |
| **For You Blue** | 60.3% | 146.1s | 60 | 53 |
| **Love Me Do** | 58.2% | 138.2s | 71 | 25 |

**WCSR Global**: 61.81% (446.9s correctas / 509.0s totales)

### 🚀 Nueva Configuración (DNN Frames)
| Canción | WCSR | Duración Total | Correcta |
|---------|------|----------------|----------|
| **Love Me Do** | 92.17% | 138.2s | 127.3s |
| **Please Please Me** | 86.46% | 119.0s | 102.9s |
| **Misery** | 80.38% | 105.8s | 85.0s |
| **For You Blue** | 57.50% | 146.1s | 84.0s |

**WCSR Global**: 78.43% (399.2s correctas / 509.0s totales)

## 🔧 **Configuraciones Técnicas**

### Pre-procesamiento (Configuración Antigua)
- **Sample Rate**: 22050 Hz
- **Chroma Features**: 12 bins (C, Db, D, Eb, E, F, Gb, G, Ab, A, Bb, B)
- **Beats per Segment**: 4
- **Normalización**: StandardScaler

### Modelo MLP (Antiguo)
- **Arquitectura**: 12 → 128 → 64 → 24
- **Activación**: ReLU + Softmax
- **Regularización**: L2 (1e-4) + Dropout (0.2)
- **Optimizador**: Adam (lr=3e-4)
- **Class Weight**: Balanced

### HMM Post-processing (Antiguo)
- **Algoritmo**: Viterbi
- **Transition Weight**: 0.3
- **Estados**: 24 clases de acordes
- **Matriz de Transición**: Aprendida del dataset

---

### 🆕 Configuración DNN Frames (Mejor Resultado)

**Pre-procesamiento**:
- **Sample Rate**: 11025 Hz
- **Hop Length**: 512 (≈46.4 ms por frame)
- **Features**: CQT (180 bins) → PCA (~120 dims, 98% var.)
- **Splicing**: Contexto ±1 frame → dims final: ~360
- **Normalización**: StandardScaler post-PCA

**Modelo DNN**:
- **Arquitectura Común**: 1024 → 1024 → 24
- **Arquitectura Bottleneck**: 1024 → 512 → 256 → 512 → 1024 → 24
- **Activación**: ReLU + BatchNorm + Dropout (0.3)
- **Regularización**: L2 (1e-4)
- **Optimizador**: Adam (lr=1e-4) - reducido para estabilidad
- **Dropout**: 0.3 (más alto que antes)
- **Batch Size**: 128
- **Early Stopping**: patience=5

**HMM Post-processing**:
- **Algoritmo**: Viterbi (log-domain)
- **Stay Probability**: 0.995 (alta permanencia)
- **Estados**: 24 clases de acordes
- **Observaciones**: Posteriors del MLP como likelihood

## 📝 **Notas y Observaciones**

1. **Más datos = Mejor rendimiento**: Agregar canciones Beatles mejoró WCSR de 50.46% a 59.97%
2. **HMM ayuda**: Post-processing con HMM mejoró WCSR a 61.81%
3. **Arquitectura simple es mejor**: Bottleneck empeoró el rendimiento
4. **Come Together inflaba métricas**: Estaba en dataset de entrenamiento
5. **WCSR es métrica clave**: Más representativa que accuracy para chord detection
6. **🆕 Frame-wise es superior**: DNN Frames mejoró WCSR de 61.81% a 78.43% (+16.62%)
   - **Features CQT** (180 bins) vs Chroma (12 bins): mejor representación espectral
   - **PCA** reduce dimensionalidad y ruido
   - **Splicing** (±1 frame) agrega contexto temporal
   - **HMM** suaviza transiciones entre acordes

## 🎯 **Próximos Experimentos**

- [ ] Probar PCA como pre-procesamiento
- [ ] Experimentar con diferentes transition_weight en HMM
- [ ] Probar time splicing (contexto temporal)
- [ ] Evaluar en más canciones independientes
- [ ] Comparar con otros datasets de acordes

---
*Última actualización: $(date)*

