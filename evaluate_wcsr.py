# evaluate_wcsr_unified.py
"""
Script unificado para evaluar WCSR de diferentes modelos en splits train/test.
Centraliza todas las evaluaciones en un solo script configurable.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import argparse
from typing import Optional, Dict, Any, Tuple, List
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

# --- Config ---
SPLIT_JSON = Path("train_test_split.json")
BEATLES_AUDIO_DIR = Path("The Beatles Annotations/audio/The Beatles")
BEATLES_CHORDLAB_DIR = Path("The Beatles Annotations/chordlab/The Beatles")

# Constantes para normalización de acordes
NOTES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
ENH = {"C#": "Db", "D#": "Eb", "F#": "Gb", "G#": "Ab", "A#": "Bb"}


def norm_label(lab: str) -> str:
    """
    Normaliza etiquetas del .lab a 25 clases:
      - 'Root:maj' o 'Root:min'
      - 'N' para no-chord
    
    Args:
        lab: etiqueta de acorde en formato original (ej: "C:maj", "Am", "N", etc.)
    
    Returns:
        etiqueta normalizada (ej: "C:maj", "A:min", "N")
    """
    if lab.upper() == "N":
        return "N"

    parts = lab.split(":")
    root = parts[0]
    rest = parts[1] if len(parts) > 1 else "maj"

    # convertir enarmónicos a bemoles
    root = ENH.get(root, root).title()

    # decidir calidad
    if "min" in rest:
        qual = "min"
    else:
        qual = "maj"

    if root not in NOTES:
        return "N"
    return f"{root}:{qual}"


def label_at_time(t, intervals):
    """
    Devuelve el label del acorde que cubre el tiempo t.
    
    Args:
        t: tiempo en segundos
        intervals: lista de tuplas [(t0, t1, label), ...]
    
    Returns:
        label del acorde que cubre el tiempo t, o "N" si no hay ninguno
    """
    for t0, t1, lab in intervals:
        if t >= t0 and t < t1:
            return lab
    return "N"


def parse_chord_label(label: str) -> Tuple[str, str]:
    """
    Parsea un label de acorde en root y quality.
    
    Args:
        label: etiqueta en formato "Root:quality" o "N"
    
    Returns:
        tuple: (root, quality) donde root es la nota y quality es "maj", "min" o "N"
    """
    if label == "N" or label.upper() == "N":
        return "N", "N"
    
    parts = label.split(":")
    if len(parts) < 2:
        return "N", "N"
    
    root = parts[0].strip()
    quality = parts[1].strip()
    
    return root, quality


def classify_error(pred_label: str, ref_label: str) -> str:
    """
    Clasifica el tipo de error entre predicción y referencia.
    
    Args:
        pred_label: etiqueta predicha
        ref_label: etiqueta de referencia
    
    Returns:
        str: tipo de error ("correct", "root_error", "quality_error", "both_error", "other")
    """
    if pred_label == ref_label:
        return "correct"
    
    pred_root, pred_qual = parse_chord_label(pred_label)
    ref_root, ref_qual = parse_chord_label(ref_label)
    
    # Si alguno es "N", es un error especial
    if pred_root == "N" or ref_root == "N":
        return "other"
    
    root_match = (pred_root == ref_root)
    qual_match = (pred_qual == ref_qual)
    
    if not root_match and qual_match:
        return "root_error"
    elif root_match and not qual_match:
        return "quality_error"
    elif not root_match and not qual_match:
        return "both_error"
    else:
        return "other"


def compute_wcsr(pred_intervals, ref_intervals, exclude_N=True, dt=0.01):
    """
    Calcula WCSR (Weighted Chord Symbol Recall) comparando intervalos predichos vs ground truth.
    
    Esta función evalúa frame por frame (con resolución dt) comparando las etiquetas
    predichas con las de referencia. El WCSR es la proporción de tiempo donde las
    etiquetas coinciden, excluyendo (opcionalmente) los segmentos marcados como "N".
    
    Args:
        pred_intervals: lista de tuplas [(t0, t1, label), ...] con predicciones
        ref_intervals: lista de tuplas [(t0, t1, label), ...] con ground truth
        exclude_N: si True, excluye intervalos con label "N" del cálculo
        dt: resolución temporal en segundos para la evaluación (default: 0.01s = 10ms)
    
    Returns:
        tuple: (wcsr, correct_duration, total_duration)
            - wcsr: Weighted Chord Symbol Recall (0-1)
            - correct_duration: duración correcta en segundos
            - total_duration: duración total evaluada en segundos
    """
    total_duration = 0.0
    correct_duration = 0.0
    
    # Obtener duración máxima
    if not pred_intervals and not ref_intervals:
        return 0.0, 0.0, 0.0
    
    max_t = 0.0
    if pred_intervals:
        max_t = max(max_t, max([t1 for _, t1, _ in pred_intervals]))
    if ref_intervals:
        max_t = max(max_t, max([t1 for _, t1, _ in ref_intervals]))
    
    if max_t <= 0:
        return 0.0, 0.0, 0.0
    
    # Evaluar frame por frame
    n_frames = int(np.ceil(max_t / dt))
    
    for i in range(n_frames):
        t = i * dt
        pred_label = label_at_time(t, pred_intervals)
        ref_label = label_at_time(t, ref_intervals)
        
        if exclude_N and ref_label == "N":
            continue
            
        total_duration += dt
        if pred_label == ref_label:
            correct_duration += dt
    
    wcsr = correct_duration / total_duration if total_duration > 0 else 0.0
    return wcsr, correct_duration, total_duration


def analyze_errors(pred_intervals, ref_intervals, exclude_N=True, dt=0.01):
    """
    Analiza los errores entre predicciones y ground truth.
    
    Args:
        pred_intervals: lista de tuplas [(t0, t1, label), ...] con predicciones
        ref_intervals: lista de tuplas [(t0, t1, label), ...] con ground truth
        exclude_N: si True, excluye intervalos con label "N" del cálculo
        dt: resolución temporal en segundos para la evaluación
    
    Returns:
        dict: diccionario con análisis de errores que incluye:
            - error_types: Counter con tipos de errores y sus duraciones
            - confusion_matrix: Counter con pares (ref, pred) y sus duraciones
            - total_duration: duración total evaluada
            - correct_duration: duración correcta
    """
    error_types = Counter()
    confusion_matrix = Counter()
    total_duration = 0.0
    correct_duration = 0.0
    
    # Obtener duración máxima
    if not pred_intervals and not ref_intervals:
        return {
            'error_types': error_types,
            'confusion_matrix': confusion_matrix,
            'total_duration': 0.0,
            'correct_duration': 0.0
        }
    
    max_t = 0.0
    if pred_intervals:
        max_t = max(max_t, max([t1 for _, t1, _ in pred_intervals]))
    if ref_intervals:
        max_t = max(max_t, max([t1 for _, t1, _ in ref_intervals]))
    
    if max_t <= 0:
        return {
            'error_types': error_types,
            'confusion_matrix': confusion_matrix,
            'total_duration': 0.0,
            'correct_duration': 0.0
        }
    
    # Evaluar frame por frame
    n_frames = int(np.ceil(max_t / dt))
    
    for i in range(n_frames):
        t = i * dt
        pred_label = label_at_time(t, pred_intervals)
        ref_label = label_at_time(t, ref_intervals)
        
        if exclude_N and ref_label == "N":
            continue
        
        total_duration += dt
        error_type = classify_error(pred_label, ref_label)
        error_types[error_type] += dt
        
        if error_type == "correct":
            correct_duration += dt
        else:
            # Registrar confusión (ref, pred)
            confusion_matrix[(ref_label, pred_label)] += dt
    
    return {
        'error_types': error_types,
        'confusion_matrix': confusion_matrix,
        'total_duration': total_duration,
        'correct_duration': correct_duration
    }


def infer_and_wcsr_frames(audio_path, lab_path, pca_path, scaler_path, model_path, labels_txt, use_hmm=True):
    """
    Realiza inferencia framewise sobre un archivo de audio y calcula WCSR contra ground truth.
    
    Esta función combina inferencia y evaluación WCSR en un solo paso para el modelo de frames.
    
    Args:
        audio_path: ruta al archivo de audio (.mp3)
        lab_path: ruta al archivo .lab con ground truth
        pca_path: ruta al archivo PCA (.joblib)
        scaler_path: ruta al archivo scaler (.joblib)
        model_path: ruta al modelo (.h5)
        labels_txt: ruta al archivo de mapeo de labels (.txt)
        use_hmm: si usar HMM para suavizar predicciones
    
    Returns:
        tuple: (wcsr, times, labels_est)
            - wcsr: Weighted Chord Symbol Recall (0-1)
            - times: array de tiempos de cada frame
            - labels_est: array de labels predichas por frame
    """
    import librosa
    import joblib
    from tensorflow.keras.models import load_model
    from inference_frames import (
        cqt_frames,
        splice,
        viterbi_log,
        build_hmm,
        merge_consecutive_labels,
        SR,
        HOP,
        CTX,
    )
    
    def load_lab_frames(path):
        """Carga archivo .lab con normalización de etiquetas."""
        rows = []
        with open(path, "r") as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 3:
                    rows.append((float(p[0]), float(p[1]), norm_label(p[2])))
        rows.sort(key=lambda x: x[0])
        return rows
    
    def compute_wcsr_frames(pred_intervals, ref_intervals, exclude_N=True):
        """
        Calcula WCSR comparando intervalos predichos vs ground truth.
        Usa resolución temporal específica para frames (HOP/SR).
        """
        dt = HOP / SR  # ~0.0464 s
        total_duration = 0.0
        correct_duration = 0.0
        
        if not pred_intervals and not ref_intervals:
            return 0.0, 0.0, 0.0
        
        max_t = 0.0
        if pred_intervals:
            max_t = max(max_t, max([t1 for _, t1, _ in pred_intervals]))
        if ref_intervals:
            max_t = max(max_t, max([t1 for _, t1, _ in ref_intervals]))
        
        if max_t <= 0:
            return 0.0, 0.0, 0.0
        
        n_frames = int(np.ceil(max_t / dt))
        
        for i in range(n_frames):
            t = i * dt
            pred_label = label_at_time(t, pred_intervals)
            ref_label = label_at_time(t, ref_intervals)
            
            if exclude_N and ref_label == "N":
                continue
                
            total_duration += dt
            if pred_label == ref_label:
                correct_duration += dt
        
        wcsr = correct_duration / total_duration if total_duration > 0 else 0.0
        return wcsr, correct_duration, total_duration
    
    # 1) Cargar audio
    y, _ = librosa.load(audio_path, sr=SR, mono=True)

    # 2) Features por frame (CQT log)
    X180, times = cqt_frames(y)

    # 3) PCA + Z-score
    pca = joblib.load(pca_path)
    scaler = joblib.load(scaler_path)
    Xp = pca.transform(X180.T)         # (T, Dp)
    Xn = scaler.transform(Xp)          # (T, Dp)

    # 4) Splicing (t-1,t,t+1)
    Xsf = splice(Xn.T, ctx=CTX)        # (T, F')

    # 5) Predicciones MLP
    model = load_model(model_path)
    P = model.predict(Xsf, verbose=0)  # (T, K) posteriors
    classes = np.loadtxt(labels_txt, dtype=str)

    # 6) HMM para suavizar (opcional)
    if use_hmm:
        pi_log, A_log = build_hmm(num_classes=P.shape[1], stay_prob=0.995)
        eps = 1e-8
        B_log = np.log(np.clip(P, eps, 1.0))
        z = viterbi_log(pi_log, A_log, B_log)
        labels_est = classes[z]
    else:
        labels_est = classes[np.argmax(P, axis=1)]

    # 7) Calcular WCSR contra ground truth
    ref_intervals = load_lab_frames(lab_path)
    pred_intervals = merge_consecutive_labels(times, labels_est)
    
    wcsr, _, _ = compute_wcsr_frames(pred_intervals, ref_intervals, exclude_N=True)
    
    return wcsr, times, labels_est


# Rutas por defecto para cada tipo de modelo
DEFAULT_PATHS = {
    "baseline_mlp": {
        "model": Path("analysis_out/baseline_mlp_model.h5"),
        "labels": Path("analysis_out/mlp_label_mapping.txt"),
        "scaler": Path("analysis_out/mlp_scaler_stats.npz"),
        "csv_dataset": Path("dataset_chords.csv"),
    },
    "frames": {
        "pca": Path("pca.joblib"),
        "scaler": Path("scaler.joblib"),
        "model": Path("analysis_out_frames/dnn_bottleneck.h5"),
        "labels": Path("analysis_out_frames/label_mapping.txt"),
    },
    "frames_beatsync": {
        "pca": Path("pca.joblib"),
        "scaler": Path("scaler.joblib"),
        "model": Path("analysis_out_frames/dnn_bottleneck.h5"),
        "labels": Path("analysis_out_frames/label_mapping.txt"),
    },
}

# Directorios de salida por defecto
DEFAULT_OUTPUT_DIRS = {
    "baseline_mlp": {
        "train": Path("outputs"),
        "test": Path("outputs_test"),
    },
    "frames": {
        "train": Path("outputs"),
        "test": Path("outputs_test"),
    },
    "frames_beatsync": {
        "train": Path("outputs_train_beatsync"),
        "test": Path("outputs_test_beatsync"),
    },
}

# Rutas CSV por defecto
DEFAULT_CSV_PATHS = {
    "baseline_mlp": {
        "train": Path("analysis_out/wcsr_train_beats_results.csv"),
        "test": Path("analysis_out/wcsr_test_beats_results.csv"),
    },
    "frames": {
        "train": Path("analysis_out_frames/wcsr_train_results.csv"),
        "test": Path("analysis_out_frames/wcsr_test_results.csv"),
    },
    "frames_beatsync": {
        "train": Path("analysis_out_frames/wcsr_train_beatsync_results.csv"),
        "test": Path("analysis_out_frames/wcsr_test_beatsync_results.csv"),
    },
}


def aggregate_error_analysis(all_analyses: List[Dict]) -> Dict:
    """
    Agrega análisis de errores de múltiples canciones.
    
    Args:
        all_analyses: lista de diccionarios retornados por analyze_errors
    
    Returns:
        dict: análisis agregado con los mismos campos que analyze_errors
    """
    aggregated_error_types = Counter()
    aggregated_confusion = Counter()
    total_duration = 0.0
    correct_duration = 0.0
    
    for analysis in all_analyses:
        aggregated_error_types.update(analysis['error_types'])
        aggregated_confusion.update(analysis['confusion_matrix'])
        total_duration += analysis['total_duration']
        correct_duration += analysis['correct_duration']
    
    return {
        'error_types': aggregated_error_types,
        'confusion_matrix': aggregated_confusion,
        'total_duration': total_duration,
        'correct_duration': correct_duration
    }


def generate_error_report(error_analysis: Dict, output_path: Path):
    """
    Genera un reporte de texto con análisis de errores.
    
    Args:
        error_analysis: diccionario retornado por aggregate_error_analysis
        output_path: ruta donde guardar el reporte
    """
    error_types = error_analysis['error_types']
    confusion_matrix = error_analysis['confusion_matrix']
    total_duration = error_analysis['total_duration']
    correct_duration = error_analysis['correct_duration']
    
    wcsr = correct_duration / total_duration if total_duration > 0 else 0.0
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("ANÁLISIS DE ERRORES - DETECCIÓN DE ACORDES")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"WCSR: {wcsr:.4f} ({wcsr*100:.2f}%)")
    report_lines.append(f"Duración total evaluada: {total_duration:.1f} segundos")
    report_lines.append(f"Duración correcta: {correct_duration:.1f} segundos")
    report_lines.append(f"Duración con errores: {total_duration - correct_duration:.1f} segundos")
    report_lines.append("")
    
    # Tipos de errores
    report_lines.append("-" * 80)
    report_lines.append("TIPOS DE ERRORES")
    report_lines.append("-" * 80)
    
    type_labels = {
        'correct': 'Correcto',
        'root_error': 'Error de Root (misma calidad, diferente nota)',
        'quality_error': 'Error de Calidad (mismo root, diferente maj/min)',
        'both_error': 'Error de Root y Calidad',
        'other': 'Otros errores'
    }
    
    for error_type in ['correct', 'root_error', 'quality_error', 'both_error', 'other']:
        duration = error_types.get(error_type, 0.0)
        percentage = (duration / total_duration * 100) if total_duration > 0 else 0.0
        report_lines.append(f"{type_labels.get(error_type, error_type):50s}: {duration:8.1f}s ({percentage:5.2f}%)")
    
    report_lines.append("")
    
    # Top confusiones
    report_lines.append("-" * 80)
    report_lines.append("TOP 20 CONFUSIONES MÁS FRECUENTES")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Referencia':<25s} -> {'Predicción':<25s} {'Duración (s)':>15s} {'%':>10s}")
    report_lines.append("-" * 80)
    
    top_confusions = confusion_matrix.most_common(20)
    for (ref, pred), duration in top_confusions:
        percentage = (duration / total_duration * 100) if total_duration > 0 else 0.0
        report_lines.append(f"{ref:<25s} -> {pred:<25s} {duration:>15.1f} {percentage:>9.2f}%")
    
    report_lines.append("")
    
    # Análisis por root
    report_lines.append("-" * 80)
    report_lines.append("ERRORES POR ROOT (nota fundamental)")
    report_lines.append("-" * 80)
    
    root_stats = defaultdict(lambda: {'correct': 0.0, 'total': 0.0})
    
    # Contar desde la matriz de confusión
    for (ref, pred), duration in confusion_matrix.items():
        ref_root, _ = parse_chord_label(ref)
        if ref_root != "N":
            root_stats[ref_root]['total'] += duration
            if ref == pred:
                root_stats[ref_root]['correct'] += duration
    
    # Agregar correctos desde error_types
    correct_dur = error_types.get('correct', 0.0)
    # Necesitamos distribuir los correctos por root. Para esto, usamos la proporción
    # de cada root en el total de referencias (sin contar correctos)
    if correct_dur > 0:
        # Calcular proporción de cada root en las referencias
        total_ref_duration = sum(root_stats[r]['total'] for r in root_stats)
        if total_ref_duration > 0:
            for root in root_stats:
                # Proporción de este root en las referencias
                proportion = root_stats[root]['total'] / total_ref_duration
                root_stats[root]['correct'] += correct_dur * proportion
                root_stats[root]['total'] += correct_dur * proportion
    
    report_lines.append(f"{'Root':<10s} {'Correcto (s)':>15s} {'Total (s)':>15s} {'WCSR':>10s}")
    report_lines.append("-" * 80)
    
    # Ordenar por WCSR descendente
    root_list = sorted(root_stats.keys(), 
                      key=lambda r: root_stats[r]['correct'] / root_stats[r]['total'] 
                      if root_stats[r]['total'] > 0 else 0.0, 
                      reverse=True)
    
    for root in root_list:
        correct = root_stats[root]['correct']
        total = root_stats[root]['total']
        root_wcsr = correct / total if total > 0 else 0.0
        report_lines.append(f"{root:<10s} {correct:>15.1f} {total:>15.1f} {root_wcsr:>9.2%}")
    
    report_lines.append("")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"📄 Reporte de errores guardado en: {output_path}")


def plot_confusion_matrix(error_analysis: Dict, output_path: Path, top_n: int = 25):
    """
    Genera visualización de la matriz de confusión (top N pares más frecuentes).
    
    Args:
        error_analysis: diccionario retornado por aggregate_error_analysis
        output_path: ruta donde guardar la imagen
        top_n: número de pares más frecuentes a mostrar
    """
    confusion_matrix = error_analysis['confusion_matrix']
    
    if len(confusion_matrix) == 0:
        print("⚠️  No hay confusiones para visualizar")
        return
    
    # Obtener top N confusiones
    top_confusions = confusion_matrix.most_common(top_n)
    
    # Crear DataFrame para visualización
    refs = [ref for (ref, pred), _ in top_confusions]
    preds = [pred for (ref, pred), _ in top_confusions]
    durations = [dur for (ref, pred), dur in top_confusions]
    
    # Crear matriz pivotada
    df = pd.DataFrame({
        'Reference': refs,
        'Prediction': preds,
        'Duration': durations
    })
    
    # Crear matriz de confusión pivotada
    pivot = df.pivot_table(values='Duration', index='Reference', columns='Prediction', fill_value=0.0)
    
    # Ordenar por frecuencia total
    ref_totals = pivot.sum(axis=1).sort_values(ascending=False)
    pred_totals = pivot.sum(axis=0).sort_values(ascending=False)
    
    pivot = pivot.reindex(index=ref_totals.index[:top_n], columns=pred_totals.index[:top_n], fill_value=0.0)
    
    # Crear visualización
    plt.figure(figsize=(14, 12))
    
    # Normalizar por filas para mejor visualización
    pivot_norm = pivot.div(pivot.sum(axis=1), axis=0).fillna(0.0)
    
    sns.heatmap(
        pivot_norm,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Proporción del tiempo'},
        xticklabels=True,
        yticklabels=True,
        linewidths=0.5
    )
    
    plt.title(f'Matriz de Confusión - Top {top_n} Confusiones\n(Proporción del tiempo por referencia)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Predicción', fontsize=12)
    plt.ylabel('Referencia (Ground Truth)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Matriz de confusión guardada en: {output_path}")


def plot_error_types(error_analysis: Dict, output_path: Path):
    """
    Genera visualización de tipos de errores.
    
    Args:
        error_analysis: diccionario retornado por aggregate_error_analysis
        output_path: ruta donde guardar la imagen
    """
    error_types = error_analysis['error_types']
    total_duration = error_analysis['total_duration']
    
    if total_duration == 0:
        print("⚠️  No hay datos para visualizar")
        return
    
    # Preparar datos
    type_labels = {
        'correct': 'Correcto',
        'root_error': 'Error de Root',
        'quality_error': 'Error de Calidad',
        'both_error': 'Error Root+Calidad',
        'other': 'Otros'
    }
    
    labels = []
    durations = []
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#95a5a6']
    
    for i, error_type in enumerate(['correct', 'root_error', 'quality_error', 'both_error', 'other']):
        duration = error_types.get(error_type, 0.0)
        if duration > 0:
            labels.append(type_labels.get(error_type, error_type))
            durations.append(duration)
    
    if len(durations) == 0:
        print("⚠️  No hay datos para visualizar")
        return
    
    # Crear gráfico
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Gráfico de barras
    bars = ax1.bar(labels, durations, color=colors[:len(labels)])
    ax1.set_ylabel('Duración (segundos)', fontsize=12)
    ax1.set_title('Distribución de Tipos de Errores', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    
    # Agregar valores en las barras
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s',
                ha='center', va='bottom', fontsize=10)
    
    # Gráfico de torta
    percentages = [(d / total_duration * 100) for d in durations]
    ax2.pie(durations, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors[:len(labels)])
    ax2.set_title('Proporción de Tipos de Errores', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Gráfico de tipos de errores guardado en: {output_path}")


def load_split_from_json(json_path=SPLIT_JSON):
    """Carga el split train/test desde train_test_split.json"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    train_songs = set(data["train_songs"])
    test_songs = set(data["test_songs"])
    return train_songs, test_songs


def load_lab_file(lab_path):
    """Carga .lab y normaliza a Root:maj/min, filtrando 'N'."""
    rows = []
    with open(lab_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                t0 = float(parts[0])
                t1 = float(parts[1])
                lab = parts[2]
                lab_n = norm_label(lab)
                if lab_n == 'N':
                    continue
                rows.append((t0, t1, lab_n))
    return rows


def evaluate_baseline_mlp(
    split: str,
    model_path: Path,
    labels_path: Path,
    scaler_path: Path,
    csv_dataset: Optional[Path],
    save_outputs_dir: Path,
    save_csv_path: Path,
    beats_per_segment: int = 4,
    use_hmm: bool = False,
    transition_weight: float = 0.3,
    max_songs: Optional[int] = None,
):
    """
    Evalúa WCSR usando el modelo baseline MLP (inferencia por beats).
    """
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    from inference import infer_on_audio, merge_consecutive_same_label

    # Verificar que el modelo existe
    if not model_path.exists():
        raise FileNotFoundError(f"No existe {model_path}")

    train_songs_set, test_songs_set = load_split_from_json(SPLIT_JSON)
    songs_set = train_songs_set if split == "train" else test_songs_set
    songs = sorted(songs_set)

    if max_songs is not None and max_songs > 0:
        songs = songs[:max_songs]

    print("\n" + "=" * 60)
    print(f"🎵 Evaluación WCSR - {split.upper()} (beats/MLP)")
    print("=" * 60)
    print(f"📊 Split desde {SPLIT_JSON.name}:")
    print(f"   Train: {len(train_songs_set)} canciones")
    print(f"   Test:  {len(test_songs_set)} canciones")
    total_songs = len(songs_set)
    if max_songs is not None:
        print(f"\nCanciones en {split.upper()}: {total_songs} (evaluando {len(songs)}/{max_songs})  | beats_per_segment={beats_per_segment} | HMM={use_hmm}")
    else:
        print(f"\nCanciones en {split.upper()}: {len(songs)}  | beats_per_segment={beats_per_segment} | HMM={use_hmm}")

    results = []
    total_correct = 0.0
    total_duration = 0.0
    wcsr_list = []

    save_outputs_dir.mkdir(exist_ok=True)

    for i, rel in enumerate(sorted(songs)):
        audio_path = BEATLES_AUDIO_DIR / f"{rel}.mp3"
        lab_path = BEATLES_CHORDLAB_DIR / f"{rel}.lab"
        if not audio_path.exists() or not lab_path.exists():
            print(f"  ⚠️  Falta audio o lab para {rel}")
            continue

        print(f"[{i+1}/{len(songs)}] {rel}")
        try:
            df_pred = infer_on_audio(
                audio_path=audio_path,
                model_path=model_path,
                labels_path=labels_path,
                scaler_stats_path=scaler_path,
                beats_per_segment=beats_per_segment,
                use_hmm=use_hmm,
                transition_weight=transition_weight,
            )

            merged = merge_consecutive_same_label(df_pred[["t_start", "t_end", "label_pred"]])
            predictions = [(t0, t1, lab) for t0, t1, lab in merged]
            ground_truth = load_lab_file(lab_path)

            wcsr, correct, total = compute_wcsr(predictions, ground_truth)

            results.append({
                'track_id': rel,
                'wcsr': wcsr,
                'correct_duration': correct,
                'total_duration': total,
                'num_predictions': len(predictions),
                'num_ground_truth': len(ground_truth),
            })

            total_correct += correct
            total_duration += total
            wcsr_list.append(wcsr)
            print(f"   WCSR={wcsr:.3f} ({correct:.1f}s / {total:.1f}s)")

            # Guardar .lab predicho
            output_file = save_outputs_dir / f"{Path(rel).name}_predicted.lab"
            with open(output_file, "w") as f:
                for t0, t1, lab in merged:
                    f.write(f"{t0:.6f} {t1:.6f} {lab}\n")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(results) == 0:
        print("\n❌ No hay resultados.")
        return None

    global_wcsr = total_correct / total_duration if total_duration > 0 else 0.0
    mean_wcsr = float(np.mean(wcsr_list)) if wcsr_list else 0.0
    median_wcsr = float(np.median(wcsr_list)) if wcsr_list else 0.0
    std_wcsr = float(np.std(wcsr_list)) if wcsr_list else 0.0

    print("\n" + "=" * 60)
    print(f"📊 RESULTADOS {split.upper()} (beats/MLP)")
    print("=" * 60)
    print(f"Canciones: {len(results)}")
    print(f"WCSR Global (ponderado): {global_wcsr:.4f} ({global_wcsr*100:.2f}%)")
    print(f"WCSR Promedio:           {mean_wcsr:.4f} ({mean_wcsr*100:.2f}%)")
    print(f"WCSR Mediana:            {median_wcsr:.4f}")
    print(f"Desv. Est.:              {std_wcsr:.4f}")

    save_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(save_csv_path, index=False)
    print(f"\n💾 Resultados guardados en: {save_csv_path}")

    return {
        'global_wcsr': global_wcsr,
        'mean_wcsr': mean_wcsr,
        'median_wcsr': median_wcsr,
        'std_wcsr': std_wcsr,
        'results': results,
        'total_correct': total_correct,
        'total_duration': total_duration,
    }


def evaluate_frames(
    split: str,
    pca_path: Path,
    scaler_path: Path,
    model_path: Path,
    labels_path: Path,
    save_outputs_dir: Path,
    save_csv_path: Path,
    use_hmm: bool = True,
    max_songs: Optional[int] = None,
    analyze_errors_flag: bool = True,
):
    """
    Evalúa WCSR usando el modelo de frames (sin beat-sync).
    
    Args:
        analyze_errors_flag: si True, genera análisis detallado de errores
    """
    from inference_frames import merge_consecutive_labels

    train_songs_set, test_songs_set = load_split_from_json(SPLIT_JSON)
    songs_set = train_songs_set if split == "train" else test_songs_set
    songs = sorted(songs_set)

    if max_songs is not None and max_songs > 0:
        songs = songs[:max_songs]

    print(f"\n📊 Split desde {SPLIT_JSON.name}:")
    print(f"   Train: {len(train_songs_set)} canciones")
    print(f"   Test:  {len(test_songs_set)} canciones")
    total_songs = len(songs_set)
    if max_songs is not None:
        print(f"\n🎵 Evaluando {split.upper()} split: {total_songs} canciones totales (evaluando {len(songs)}/{max_songs})...\n")
    else:
        print(f"\n🎵 Evaluando {split.upper()} split (canciones: {len(songs)})...\n")

    save_outputs_dir.mkdir(exist_ok=True)

    results = []
    n_songs = 0
    wcsr_list = []
    error_analyses = []  # Para acumular análisis de errores

    for rel in sorted(songs):
        audio_path = BEATLES_AUDIO_DIR / f"{rel}.mp3"
        lab_path = BEATLES_CHORDLAB_DIR / f"{rel}.lab"

        if not lab_path.exists():
            print(f"  ⚠️  Lab faltante: {lab_path}")
            continue
        if not audio_path.exists():
            print(f"  ⚠️  Audio faltante: {audio_path}")
            continue

        print(f"▶ {rel}")
        try:
            w, t, L = infer_and_wcsr_frames(audio_path, lab_path, pca_path, scaler_path, model_path, labels_path, use_hmm=use_hmm)
            n_songs += 1
            wcsr_list.append(w)

            results.append({
                'song': rel,
                'wcsr': w,
            })

            print(f"  WCSR: {w:.4f} ({w*100:.2f}%)\n")

            # Guardar .lab predicho
            intervals = merge_consecutive_labels(t, L)
            output_file = save_outputs_dir / f"{Path(rel).name}_predicted.lab"
            with open(output_file, "w") as f:
                for t0, t1, lab in intervals:
                    f.write(f"{t0:.6f} {t1:.6f} {lab}\n")
            
            # Análisis de errores por canción
            if analyze_errors_flag:
                from inference_frames import HOP, SR
                ref_intervals = load_lab_file(lab_path)
                error_analysis = analyze_errors(intervals, ref_intervals, exclude_N=True, dt=HOP/SR)
                error_analyses.append(error_analysis)

        except Exception as e:
            print(f"  ❌ Error: {e}\n")
            import traceback
            traceback.print_exc()
            continue

    if n_songs > 0:
        mean_wcsr = float(np.mean(wcsr_list)) if wcsr_list else 0.0
        median_wcsr = float(np.median(wcsr_list)) if wcsr_list else 0.0
        std_wcsr = float(np.std(wcsr_list)) if wcsr_list else 0.0

        print(f"\n{'='*60}")
        print(f"📊 RESULTADOS {split.upper()} (por canción)")
        print(f"{'='*60}")
        print(f"\nResultados por canción:")
        for i, r in enumerate(sorted(results, key=lambda x: x['wcsr'], reverse=True), 1):
            print(f"  {i}. {r['song']:40s}  WCSR={r['wcsr']:.4f} ({r['wcsr']*100:.2f}%)")
        print(f"\nCanciones ({split}): {n_songs}")
        print(f"WCSR Promedio: {mean_wcsr:.4f} ({mean_wcsr*100:.2f}%)")
        print(f"WCSR Mediana:  {median_wcsr:.4f}")
        print(f"Desv. Est.:    {std_wcsr:.4f}")
        print(f"\n{'='*60}\n")

        save_csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(save_csv_path, index=False)
        print(f"💾 Resultados guardados en: {save_csv_path}")
        
        # Generar análisis de errores
        if analyze_errors_flag and len(error_analyses) > 0:
            print("\n" + "=" * 60)
            print("🔍 GENERANDO ANÁLISIS DE ERRORES...")
            print("=" * 60)
            
            aggregated_analysis = aggregate_error_analysis(error_analyses)
            
            # Rutas de salida para reportes
            error_report_path = save_csv_path.parent / f"error_analysis_{split}.txt"
            confusion_matrix_path = save_csv_path.parent / f"confusion_matrix_{split}.png"
            error_types_path = save_csv_path.parent / f"error_types_{split}.png"
            
            # Generar reportes y visualizaciones
            generate_error_report(aggregated_analysis, error_report_path)
            plot_confusion_matrix(aggregated_analysis, confusion_matrix_path, top_n=25)
            plot_error_types(aggregated_analysis, error_types_path)
            
            print("\n✅ Análisis de errores completado!")

    return results


def evaluate_frames_beatsync(
    split: str,
    pca_path: Path,
    scaler_path: Path,
    model_path: Path,
    labels_path: Path,
    save_outputs_dir: Path,
    save_csv_path: Path,
    use_hmm: bool = True,
    beat_group: int = 1,
    max_songs: Optional[int] = None,
    analyze_errors_flag: bool = True,
):
    """
    Evalúa WCSR usando el modelo de frames con beat-sync.
    
    Args:
        analyze_errors_flag: si True, genera análisis detallado de errores
    """
    from inference_frames import infer_on_audio, norm_label, SR, HOP

    def load_lab(path):
        """Carga archivo .lab con normalización."""
        rows = []
        with open(path, "r") as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 3:
                    rows.append((float(p[0]), float(p[1]), norm_label(p[2])))
        rows.sort(key=lambda x: x[0])
        return rows

    def label_at_time(t, intervals):
        """Devuelve el label del acorde que cubre el tiempo t."""
        for t0, t1, lab in intervals:
            if t >= t0 and t < t1:
                return lab
        return "N"

    def compute_wcsr_local(pred_intervals, ref_intervals, exclude_N=True):
        """Calcula WCSR comparando intervalos predichos vs ground truth."""
        dt = HOP / SR  # ~0.0464 s
        total_duration = 0.0
        correct_duration = 0.0

        if not pred_intervals and not ref_intervals:
            return 0.0, 0.0, 0.0

        max_t = 0.0
        if pred_intervals:
            max_t = max(max_t, max([t1 for _, t1, _ in pred_intervals]))
        if ref_intervals:
            max_t = max(max_t, max([t1 for _, t1, _ in ref_intervals]))

        if max_t <= 0:
            return 0.0, 0.0, 0.0

        n_frames = int(np.ceil(max_t / dt))

        for i in range(n_frames):
            t = i * dt
            pred_label = label_at_time(t, pred_intervals)
            ref_label = label_at_time(t, ref_intervals)

            if exclude_N and ref_label == "N":
                continue

            total_duration += dt
            if pred_label == ref_label:
                correct_duration += dt

        wcsr = correct_duration / total_duration if total_duration > 0 else 0.0
        return wcsr, correct_duration, total_duration

    train_songs_set, test_songs_set = load_split_from_json(SPLIT_JSON)
    songs_set = train_songs_set if split == "train" else test_songs_set
    songs = sorted(songs_set)

    if max_songs is not None and max_songs > 0:
        songs = songs[:max_songs]

    print(f"\n📊 Split desde {SPLIT_JSON.name}:")
    print(f"   Train: {len(train_songs_set)} canciones")
    print(f"   Test:  {len(test_songs_set)} canciones")
    print(f"\n🎵 Evaluando {split.upper()} split con BEAT-SYNC:")
    print(f"   - HMM: {use_hmm}")
    print(f"   - Beat group: {beat_group}")
    total_songs = len(songs_set)
    if max_songs is not None:
        print(f"   - Canciones: {total_songs} totales (evaluando {len(songs)}/{max_songs})...\n")
    else:
        print(f"   - Canciones: {len(songs)}...\n")

    save_outputs_dir.mkdir(exist_ok=True)

    results = []
    n_songs = 0
    wcsr_list = []
    error_analyses = []  # Para acumular análisis de errores

    for rel in sorted(songs):
        audio_path = BEATLES_AUDIO_DIR / f"{rel}.mp3"
        lab_path = BEATLES_CHORDLAB_DIR / f"{rel}.lab"

        if not lab_path.exists():
            print(f"  ⚠️  Lab faltante: {lab_path}")
            continue
        if not audio_path.exists():
            print(f"  ⚠️  Audio faltante: {audio_path}")
            continue

        print(f"▶ {rel}")
        try:
            # Inferencia con beat-sync
            times, labels_est, intervals = infer_on_audio(
                audio_path=audio_path,
                pca_path=pca_path,
                scaler_path=scaler_path,
                model_path=model_path,
                labels_txt=labels_path,
                use_hmm=use_hmm,
                beat_sync=True,
                beat_group=beat_group
            )

            # Cargar ground truth
            ref_intervals = load_lab(lab_path)

            # Calcular WCSR
            wcsr, correct_dur, total_dur = compute_wcsr_local(intervals, ref_intervals, exclude_N=True)

            n_songs += 1
            wcsr_list.append(wcsr)

            results.append({
                'song': rel,
                'wcsr': wcsr,
                'correct_duration': correct_dur,
                'total_duration': total_dur,
            })

            print(f"  WCSR: {wcsr:.4f} ({wcsr*100:.2f}%)")
            print(f"  Correct: {correct_dur:.1f}s / Total: {total_dur:.1f}s\n")

            # Guardar .lab predicho
            output_file = save_outputs_dir / f"{Path(rel).name}_predicted.lab"
            with open(output_file, "w") as f:
                for t0, t1, lab in intervals:
                    f.write(f"{t0:.6f} {t1:.6f} {lab}\n")
            
            # Análisis de errores por canción
            if analyze_errors_flag:
                error_analysis = analyze_errors(intervals, ref_intervals, exclude_N=True, dt=HOP/SR)
                error_analyses.append(error_analysis)

        except Exception as e:
            print(f"  ❌ Error: {e}\n")
            import traceback
            traceback.print_exc()
            continue

    if n_songs > 0:
        mean_wcsr = float(np.mean(wcsr_list)) if wcsr_list else 0.0
        median_wcsr = float(np.median(wcsr_list)) if wcsr_list else 0.0
        std_wcsr = float(np.std(wcsr_list)) if wcsr_list else 0.0

        # Calcular WCSR global (ponderado por duración)
        total_correct = sum([r['correct_duration'] for r in results])
        total_duration = sum([r['total_duration'] for r in results])
        global_wcsr = total_correct / total_duration if total_duration > 0 else 0.0

        print(f"\n{'='*60}")
        print(f"📊 RESULTADOS {split.upper()} (BEAT-SYNC)")
        print(f"{'='*60}")
        print(f"\nResultados por canción:")
        for i, r in enumerate(sorted(results, key=lambda x: x['wcsr'], reverse=True), 1):
            print(f"  {i}. {r['song']:40s}  WCSR={r['wcsr']:.4f} ({r['wcsr']*100:.2f}%)")
        print(f"\nCanciones ({split}): {n_songs}")
        print(f"WCSR Global (ponderado): {global_wcsr:.4f} ({global_wcsr*100:.2f}%)")
        print(f"WCSR Promedio:           {mean_wcsr:.4f} ({mean_wcsr*100:.2f}%)")
        print(f"WCSR Mediana:            {median_wcsr:.4f}")
        print(f"Desv. Est.:              {std_wcsr:.4f}")
        print(f"\n{'='*60}\n")

        save_csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(save_csv_path, index=False)
        print(f"💾 Resultados guardados en: {save_csv_path}")
        
        # Generar análisis de errores
        if analyze_errors_flag and len(error_analyses) > 0:
            print("\n" + "=" * 60)
            print("🔍 GENERANDO ANÁLISIS DE ERRORES...")
            print("=" * 60)
            
            aggregated_analysis = aggregate_error_analysis(error_analyses)
            
            # Rutas de salida para reportes
            error_report_path = save_csv_path.parent / f"error_analysis_{split}_beatsync.txt"
            confusion_matrix_path = save_csv_path.parent / f"confusion_matrix_{split}_beatsync.png"
            error_types_path = save_csv_path.parent / f"error_types_{split}_beatsync.png"
            
            # Generar reportes y visualizaciones
            generate_error_report(aggregated_analysis, error_report_path)
            plot_confusion_matrix(aggregated_analysis, confusion_matrix_path, top_n=25)
            plot_error_types(aggregated_analysis, error_types_path)
            
            print("\n✅ Análisis de errores completado!")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluación unificada de WCSR para diferentes modelos y splits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Evaluar baseline MLP en test split
  python evaluate_wcsr_unified.py baseline_mlp test

  # Evaluar frames model en train split
  python evaluate_wcsr_unified.py frames train

  # Evaluar frames con beat-sync en test split
  python evaluate_wcsr_unified.py frames_beatsync test --beat-group 4

  # Evaluar con rutas personalizadas
  python evaluate_wcsr_unified.py frames test \\
    --model-path analysis_out_frames/dnn_common.h5 \\
    --pca-path pca.joblib \\
    --scaler-path scaler.joblib \\
    --labels-path analysis_out_frames/label_mapping.txt

  # Limitar cantidad de canciones para prueba rápida
  python evaluate_wcsr_unified.py baseline_mlp test --max-songs 10
        """
    )

    parser.add_argument(
        'model_type',
        choices=['baseline_mlp', 'frames', 'frames_beatsync'],
        help='Tipo de modelo a evaluar'
    )

    parser.add_argument(
        'split',
        choices=['train', 'test'],
        help='Split a evaluar (train o test)'
    )

    # Parámetros comunes
    parser.add_argument('--max-songs', type=int, default=None,
                       help='Límite de canciones a evaluar (None = todas)')

    # Parámetros específicos de baseline_mlp
    parser.add_argument('--beats-per-segment', type=int, default=4,
                       help='Número de beats por segmento (solo baseline_mlp)')
    parser.add_argument('--use-hmm-baseline', action='store_true',
                       help='Usar HMM para suavizar (solo baseline_mlp)')
    parser.add_argument('--transition-weight', type=float, default=0.3,
                       help='Peso de transiciones HMM (solo baseline_mlp)')

    # Parámetros específicos de frames
    parser.add_argument('--use-hmm-frames', action='store_true', default=True,
                       help='Usar HMM para suavizar (solo frames, default=True)')
    parser.add_argument('--no-hmm-frames', dest='use_hmm_frames', action='store_false',
                       help='No usar HMM (solo frames)')

    # Parámetros específicos de frames_beatsync
    parser.add_argument('--beat-group', type=int, default=2,
                       help='Cantidad de beats por grupo para majority voting (solo frames_beatsync)')

    # Rutas de modelos (opcionales, con defaults)
    parser.add_argument('--model-path', type=Path, default=None,
                       help='Ruta al modelo (.h5 o .pkl)')
    parser.add_argument('--labels-path', type=Path, default=None,
                       help='Ruta al archivo de mapeo de labels (.txt)')
    parser.add_argument('--scaler-path', type=Path, default=None,
                       help='Ruta al scaler (.joblib o .npz)')
    parser.add_argument('--pca-path', type=Path, default=None,
                       help='Ruta al PCA (.joblib, solo frames)')
    parser.add_argument('--csv-dataset', type=Path, default=None,
                       help='Ruta al CSV dataset (solo baseline_mlp)')

    # Rutas de salida (opcionales)
    parser.add_argument('--output-dir', type=Path, default=None,
                       help='Directorio donde guardar los .lab predichos')
    parser.add_argument('--csv-output', type=Path, default=None,
                       help='Ruta donde guardar el CSV con resultados')

    args = parser.parse_args()

    # Obtener rutas por defecto según el tipo de modelo
    defaults = DEFAULT_PATHS[args.model_type]
    split = args.split

    # Resolver rutas de modelo
    if args.model_type == "baseline_mlp":
        model_path = args.model_path or defaults["model"]
        labels_path = args.labels_path or defaults["labels"]
        scaler_path = args.scaler_path or defaults["scaler"]
        csv_dataset = args.csv_dataset or defaults["csv_dataset"]
        pca_path = None
    else:  # frames o frames_beatsync
        pca_path = args.pca_path or defaults["pca"]
        scaler_path = args.scaler_path or defaults["scaler"]
        model_path = args.model_path or defaults["model"]
        labels_path = args.labels_path or defaults["labels"]
        csv_dataset = None

    # Resolver rutas de salida
    output_dir = args.output_dir or DEFAULT_OUTPUT_DIRS[args.model_type][split]
    csv_output = args.csv_output or DEFAULT_CSV_PATHS[args.model_type][split]

    # Ejecutar evaluación según el tipo de modelo
    if args.model_type == "baseline_mlp":
        evaluate_baseline_mlp(
            split=split,
            model_path=model_path,
            labels_path=labels_path,
            scaler_path=scaler_path,
            csv_dataset=csv_dataset,
            save_outputs_dir=output_dir,
            save_csv_path=csv_output,
            beats_per_segment=args.beats_per_segment,
            use_hmm=args.use_hmm_baseline,
            transition_weight=args.transition_weight,
            max_songs=args.max_songs,
        )
    elif args.model_type == "frames":
        evaluate_frames(
            split=split,
            pca_path=pca_path,
            scaler_path=scaler_path,
            model_path=model_path,
            labels_path=labels_path,
            save_outputs_dir=output_dir,
            save_csv_path=csv_output,
            use_hmm=args.use_hmm_frames,
            max_songs=args.max_songs,
        )
    elif args.model_type == "frames_beatsync":
        evaluate_frames_beatsync(
            split=split,
            pca_path=pca_path,
            scaler_path=scaler_path,
            model_path=model_path,
            labels_path=labels_path,
            save_outputs_dir=output_dir,
            save_csv_path=csv_output,
            use_hmm=args.use_hmm_frames,
            beat_group=args.beat_group,
            max_songs=args.max_songs,
        )


if __name__ == "__main__":
    main()

