# evaluate_wcsr.py
"""
Evaluación end-to-end usando WCSR (Weighted Chord Symbol Recall).

Mide el sistema COMPLETO: audio → preprocesamiento → MLP → predicción
comparado contra ground truth annotations.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

# Importar función de inferencia
from inference import infer_on_audio, merge_consecutive_same_label

# Rutas
BEATLES_AUDIO_DIR = Path("The Beatles Annotations/audio/The Beatles")
BEATLES_CHORDLAB_DIR = Path("The Beatles Annotations/chordlab/The Beatles")
OUT = Path("analysis_out")

RANDOM_STATE = 42

# Normalización de labels (mismo código que build_dataset.py)
NOTES = ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"]
ENH = {"C#":"Db","D#":"Eb","F#":"Gb","G#":"Ab","A#":"Bb"}

def norm_label(lab: str) -> str:
    """
    Normaliza etiquetas a formato Root:maj o Root:min.
    Quita extensiones como 7, maj7, sus4, etc.
    
    Ejemplos:
        'C:7' → 'C:maj'
        'C:min7' → 'C:min'
        'C#:maj' → 'Db:maj'
        'N' → 'N' (no-chord)
    """
    if lab.upper() == "N":
        return "N"

    parts = lab.split(":")
    root = parts[0]
    rest = parts[1] if len(parts) > 1 else "maj"

    # Convertir enarmónicos a bemoles
    root = ENH.get(root, root).title()

    # Decidir calidad (maj o min)
    if "min" in rest:
        qual = "min"
    else:
        qual = "maj"

    if root not in NOTES:
        return "N"
    
    return f"{root}:{qual}"


def load_lab_file(lab_path, normalize=True):
    """
    Carga un archivo .lab con anotaciones de acordes.
    
    Args:
        lab_path: path al archivo .lab
        normalize: si True, normaliza labels a Root:maj/min
    
    Returns:
        list of (start, end, label) tuples
    """
    annotations = []
    with open(lab_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                start = float(parts[0])
                end = float(parts[1])
                label = parts[2]
                
                # Normalizar label
                if normalize:
                    label = norm_label(label)
                    # Filtrar 'N' (sin acorde)
                    if label == 'N':
                        continue
                
                annotations.append((start, end, label))
    return annotations


def compute_overlap(seg1, seg2):
    """
    Calcula el overlap (duración) entre dos segmentos.
    
    Args:
        seg1, seg2: (start, end) tuples
    
    Returns:
        overlap duration (0 si no hay overlap)
    """
    start = max(seg1[0], seg2[0])
    end = min(seg1[1], seg2[1])
    return max(0, end - start)


def compute_wcsr(predictions, ground_truth):
    """
    Calcula WCSR (Weighted Chord Symbol Recall).
    
    WCSR = suma(duraciones correctas) / suma(duraciones totales)
    
    Args:
        predictions: list of (start, end, label)
        ground_truth: list of (start, end, label)
    
    Returns:
        wcsr: float between 0 and 1
        correct_duration: total duration matched
        total_duration: total duration in ground truth
    """
    correct_duration = 0.0
    total_duration = 0.0
    
    for gt_start, gt_end, gt_label in ground_truth:
        gt_duration = gt_end - gt_start
        total_duration += gt_duration
        
        # Encontrar todas las predicciones que overlappean con este segmento GT
        for pred_start, pred_end, pred_label in predictions:
            overlap = compute_overlap((gt_start, gt_end), (pred_start, pred_end))
            
            if overlap > 0 and pred_label == gt_label:
                correct_duration += overlap
    
    wcsr = correct_duration / total_duration if total_duration > 0 else 0.0
    return wcsr, correct_duration, total_duration


def find_matching_audio_and_annotations():
    """
    Encuentra pares de (audio, annotation) disponibles.
    
    Returns:
        list of (audio_path, lab_path, track_id) tuples
    """
    pairs = []
    
    # Buscar todos los archivos .lab en chordlab
    for lab_file in BEATLES_CHORDLAB_DIR.rglob("*.lab"):
        # Extraer album y track
        relative = lab_file.relative_to(BEATLES_CHORDLAB_DIR)
        album = relative.parts[0]  # ej: "01_-_Please_Please_Me"
        track_name = lab_file.stem  # ej: "01_-_I_Saw_Her_Standing_There"
        
        # Buscar audio correspondiente
        audio_path = BEATLES_AUDIO_DIR / album / f"{track_name}.mp3"
        
        if audio_path.exists():
            track_id = f"{album}/{track_name}"
            pairs.append((audio_path, lab_file, track_id))
    
    return pairs


def evaluate_system_wcsr(use_hmm=False, transition_weight=0.3, beats_per_segment=4):
    """
    Evalúa el sistema completo end-to-end usando WCSR.
    
    Args:
        use_hmm: si usar HMM post-processing
        transition_weight: peso del HMM
        beats_per_segment: cuántos beats agrupar
    
    Returns:
        dict con resultados
    """
    print("=" * 60)
    print("🎵 Evaluación End-to-End con WCSR")
    print("=" * 60)
    
    # Encontrar pares audio-annotation
    pairs = find_matching_audio_and_annotations()
    print(f"\n📁 Archivos encontrados: {len(pairs)} canciones con audio + annotations")
    
    if len(pairs) == 0:
        print("❌ No se encontraron pares de audio/annotations")
        return None
    
    # Mostrar algunas canciones
    print(f"\nEjemplos:")
    for i, (audio, lab, track_id) in enumerate(pairs[:3]):
        print(f"  {i+1}. {track_id}")
    if len(pairs) > 3:
        print(f"  ... y {len(pairs) - 3} más")
    
    # Evaluar cada canción
    results = []
    total_correct = 0.0
    total_duration = 0.0
    
    print(f"\n🔄 Procesando canciones...")
    print(f"   Parámetros: beats_per_segment={beats_per_segment}, use_hmm={use_hmm}")
    
    for i, (audio_path, lab_path, track_id) in enumerate(pairs):
        print(f"\n[{i+1}/{len(pairs)}] {track_id}")
        
        try:
            # 1. Ejecutar inferencia (sistema completo)
            df_pred = infer_on_audio(
                audio_path=audio_path,
                beats_per_segment=beats_per_segment,
                use_hmm=use_hmm,
                transition_weight=transition_weight
            )
            
            # 2. Merge consecutivos y convertir a lista (ya normalizados)
            merged = merge_consecutive_same_label(df_pred[["t_start", "t_end", "label_pred"]])
            predictions = [(t0, t1, label) for t0, t1, label in merged]
            
            # 3. Cargar ground truth (con normalización)
            ground_truth = load_lab_file(lab_path, normalize=True)
            
            # 4. Calcular WCSR
            wcsr, correct, total = compute_wcsr(predictions, ground_truth)
            
            results.append({
                'track_id': track_id,
                'wcsr': wcsr,
                'correct_duration': correct,
                'total_duration': total,
                'num_predictions': len(predictions),
                'num_ground_truth': len(ground_truth)
            })
            
            total_correct += correct
            total_duration += total
            
            print(f"   WCSR: {wcsr:.3f} ({correct:.1f}s / {total:.1f}s)")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    # Resultados finales
    print("\n" + "=" * 60)
    print("📊 RESULTADOS FINALES")
    print("=" * 60)
    
    if len(results) == 0:
        print("❌ No se pudo evaluar ninguna canción")
        return None
    
    # WCSR global (ponderado por duración)
    global_wcsr = total_correct / total_duration if total_duration > 0 else 0.0
    
    # WCSR promedio (no ponderado)
    mean_wcsr = np.mean([r['wcsr'] for r in results])
    median_wcsr = np.median([r['wcsr'] for r in results])
    std_wcsr = np.std([r['wcsr'] for r in results])
    
    print(f"\nCanciones evaluadas: {len(results)}")
    print(f"\nWCSR Global (ponderado):  {global_wcsr:.4f} ({global_wcsr*100:.2f}%)")
    print(f"WCSR Promedio:            {mean_wcsr:.4f} ({mean_wcsr*100:.2f}%)")
    print(f"WCSR Mediana:             {median_wcsr:.4f}")
    print(f"Desviación estándar:      {std_wcsr:.4f}")
    
    print(f"\nDuración total evaluada:  {total_duration:.1f} segundos ({total_duration/60:.1f} minutos)")
    print(f"Duración correcta:        {total_correct:.1f} segundos ({total_correct/60:.1f} minutos)")
    
    # Top/worst canciones
    results_sorted = sorted(results, key=lambda x: x['wcsr'], reverse=True)
    
    print(f"\n🏆 Top 5 mejores:")
    for i, r in enumerate(results_sorted[:5]):
        print(f"   {i+1}. {r['track_id']:50s}  WCSR={r['wcsr']:.3f}")
    
    print(f"\n📉 Top 5 peores:")
    for i, r in enumerate(results_sorted[-5:]):
        print(f"   {i+1}. {r['track_id']:50s}  WCSR={r['wcsr']:.3f}")
    
    return {
        'global_wcsr': global_wcsr,
        'mean_wcsr': mean_wcsr,
        'median_wcsr': median_wcsr,
        'std_wcsr': std_wcsr,
        'results': results,
        'total_correct': total_correct,
        'total_duration': total_duration
    }


def visualize_wcsr_results(results_dict, output_path):
    """Visualiza la distribución de WCSR."""
    if results_dict is None:
        return
    
    results = results_dict['results']
    wcsr_values = [r['wcsr'] for r in results]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histograma
    ax = axes[0]
    ax.hist(wcsr_values, bins=20, color='#4ECDC4', alpha=0.7, edgecolor='black')
    ax.axvline(results_dict['global_wcsr'], color='red', linestyle='--', 
               linewidth=2, label=f'WCSR Global: {results_dict["global_wcsr"]:.3f}')
    ax.axvline(results_dict['mean_wcsr'], color='orange', linestyle='--', 
               linewidth=2, label=f'Media: {results_dict["mean_wcsr"]:.3f}')
    ax.set_xlabel('WCSR')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Distribución de WCSR por Canción')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Box plot
    ax = axes[1]
    bp = ax.boxplot([wcsr_values], vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('#4ECDC4')
    bp['boxes'][0].set_alpha(0.7)
    ax.set_ylabel('WCSR')
    ax.set_xticklabels(['Todas las canciones'])
    ax.set_title('Box Plot de WCSR')
    ax.grid(axis='y', alpha=0.3)
    
    # Anotar estadísticas
    ax.text(1.15, results_dict['mean_wcsr'], 
            f"Media: {results_dict['mean_wcsr']:.3f}\n" +
            f"Mediana: {results_dict['median_wcsr']:.3f}\n" +
            f"Std: {results_dict['std_wcsr']:.3f}",
            fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Evaluación End-to-End con WCSR', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\n📊 Gráfico guardado en: {output_path}")


# ==================== MAIN ====================

if __name__ == "__main__":
    print("\n🎵 WCSR Evaluation - Sistema End-to-End\n")
    
    # Evaluar sistema
    results = evaluate_system_wcsr(
        use_hmm=False,           # Cambiar a True para probar con HMM
        transition_weight=0.3,
        beats_per_segment=2      # Cambiado a 2 beats para mejor resolución temporal
    )
    
    if results is not None:
        # Guardar resultados
        results_df = pd.DataFrame(results['results'])
        results_df.to_csv(OUT / "wcsr_results.csv", index=False)
        print(f"\n💾 Resultados guardados en: {OUT / 'wcsr_results.csv'}")
        
        # Visualizar
        visualize_wcsr_results(results, OUT / "wcsr_distribution.png")
        
        # Guardar resumen
        summary = f"""WCSR Evaluation Summary
{'='*60}

System Configuration:
  - Beats per segment: 2
  - HMM: False
  
Results:
  - Songs evaluated: {len(results['results'])}
  - Global WCSR (weighted): {results['global_wcsr']:.4f} ({results['global_wcsr']*100:.2f}%)
  - Mean WCSR: {results['mean_wcsr']:.4f}
  - Median WCSR: {results['median_wcsr']:.4f}
  - Std Dev: {results['std_wcsr']:.4f}
  
Total Duration:
  - Total: {results['total_duration']:.1f}s ({results['total_duration']/60:.1f} min)
  - Correct: {results['total_correct']:.1f}s ({results['total_correct']/60:.1f} min)
  - Error: {results['total_duration'] - results['total_correct']:.1f}s
"""
        
        (OUT / "wcsr_summary.txt").write_text(summary)
        print(f"💾 Resumen guardado en: {OUT / 'wcsr_summary.txt'}")
    
    print("\n✅ Evaluación completada!")

