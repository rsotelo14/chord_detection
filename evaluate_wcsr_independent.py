# evaluate_wcsr_independent.py
"""
Evaluación WCSR en canciones INDEPENDIENTES (no usadas en entrenamiento).
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
from pathlib import Path

# Importar funciones del script original
from evaluate_wcsr import (
    compute_wcsr, 
    norm_label,
    visualize_wcsr_results
)
from inference import infer_on_audio, merge_consecutive_same_label

# Rutas
WCSR_TEST_DIR = Path("wcsr_test")
BEATLES_CHORDLAB_DIR = Path("The Beatles Annotations/chordlab/The Beatles")
OUT = Path("analysis_out")

# Mapeo manual de canciones descargadas a sus archivos .lab
SONG_MAPPINGS = {
    "For you blue.mp3": "12_-_Let_It_Be/11_-_For_You_Blue.lab",
    "Love Me Do.mp3": "01_-_Please_Please_Me/08_-_Love_Me_Do.lab",
    "Misery.mp3": "01_-_Please_Please_Me/02_-_Misery.lab",
    "Please Please Me.mp3": "01_-_Please_Please_Me/07_-_Please_Please_Me.lab",
}


def load_lab_file(lab_path, normalize=True):
    """Carga archivo .lab con normalización."""
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
                
                if normalize:
                    label = norm_label(label)
                    if label == 'N':
                        continue
                
                annotations.append((start, end, label))
    return annotations


def evaluate_independent_set():
    """Evalúa WCSR en el conjunto independiente."""
    print("=" * 60)
    print("🎵 Evaluación WCSR - Conjunto Independiente")
    print("=" * 60)
    
    # Verificar que existan los archivos
    audio_files = list(WCSR_TEST_DIR.glob("*.mp3"))
    print(f"\n📁 Canciones en wcsr_test/: {len(audio_files)}")
    
    if len(audio_files) == 0:
        print("❌ No se encontraron archivos .mp3 en wcsr_test/")
        return None
    
    for audio in audio_files:
        print(f"  - {audio.name}")
    
    # Evaluar cada canción
    results = []
    total_correct = 0.0
    total_duration = 0.0
    
    print(f"\n🔄 Procesando canciones (beats_per_segment=4)...")
    
    for audio_file in audio_files:
        audio_name = audio_file.name
        
        # Verificar si tenemos el mapeo
        if audio_name not in SONG_MAPPINGS:
            print(f"\n❌ No se encontró mapeo para: {audio_name}")
            continue
        
        lab_relative = SONG_MAPPINGS[audio_name]
        lab_path = BEATLES_CHORDLAB_DIR / lab_relative
        
        if not lab_path.exists():
            print(f"\n❌ No existe archivo .lab: {lab_path}")
            continue
        
        print(f"\n▶ {audio_name}")
        print(f"  .lab: {lab_relative}")
        
        try:
            # 1. Inferencia (sistema completo)
            df_pred = infer_on_audio(
                audio_path=audio_file,
                beats_per_segment=4,
                use_hmm=True,
                transition_weight=0.3
            )
            
            # 2. Merge y convertir a lista
            merged = merge_consecutive_same_label(df_pred[["t_start", "t_end", "label_pred"]])
            predictions = [(t0, t1, label) for t0, t1, label in merged]
            
            # 3. Cargar ground truth (normalizado)
            ground_truth = load_lab_file(lab_path, normalize=True)
            
            # 4. Calcular WCSR
            wcsr, correct, total = compute_wcsr(predictions, ground_truth)
            
            results.append({
                'track_id': audio_file.stem,
                'wcsr': wcsr,
                'correct_duration': correct,
                'total_duration': total,
                'num_predictions': len(predictions),
                'num_ground_truth': len(ground_truth)
            })
            
            total_correct += correct
            total_duration += total
            
            print(f"  WCSR: {wcsr:.3f} ({correct:.1f}s / {total:.1f}s)")
            print(f"  Predicciones: {len(predictions)} segmentos")
            print(f"  Ground truth: {len(ground_truth)} segmentos")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Resultados finales
    print("\n" + "=" * 60)
    print("📊 RESULTADOS FINALES - CONJUNTO INDEPENDIENTE")
    print("=" * 60)
    
    if len(results) == 0:
        print("❌ No se pudo evaluar ninguna canción")
        return None
    
    # WCSR global
    global_wcsr = total_correct / total_duration if total_duration > 0 else 0.0
    mean_wcsr = np.mean([r['wcsr'] for r in results])
    median_wcsr = np.median([r['wcsr'] for r in results])
    std_wcsr = np.std([r['wcsr'] for r in results])
    
    print(f"\nCanciones evaluadas: {len(results)}")
    print(f"\n🎯 WCSR Global (ponderado):  {global_wcsr:.4f} ({global_wcsr*100:.2f}%)")
    print(f"   WCSR Promedio:            {mean_wcsr:.4f} ({mean_wcsr*100:.2f}%)")
    print(f"   WCSR Mediana:             {median_wcsr:.4f}")
    print(f"   Desviación estándar:      {std_wcsr:.4f}")
    
    print(f"\nDuración total evaluada:  {total_duration:.1f} segundos ({total_duration/60:.1f} minutos)")
    print(f"Duración correcta:        {total_correct:.1f} segundos ({total_correct/60:.1f} minutos)")
    
    # Resultados por canción
    print(f"\n📋 Resultados por canción:")
    results_sorted = sorted(results, key=lambda x: x['wcsr'], reverse=True)
    for i, r in enumerate(results_sorted):
        print(f"   {i+1}. {r['track_id']:30s}  WCSR={r['wcsr']:.3f}")
    
    return {
        'global_wcsr': global_wcsr,
        'mean_wcsr': mean_wcsr,
        'median_wcsr': median_wcsr,
        'std_wcsr': std_wcsr,
        'results': results,
        'total_correct': total_correct,
        'total_duration': total_duration
    }


if __name__ == "__main__":
    print("\n🎵 Evaluación WCSR - Canciones Independientes\n")
    
    results = evaluate_independent_set()
    
    if results is not None:
        # Guardar resultados
        results_df = pd.DataFrame(results['results'])
        results_df.to_csv(OUT / "wcsr_independent_results.csv", index=False)
        print(f"\n💾 Resultados guardados en: {OUT / 'wcsr_independent_results.csv'}")
        
        # Guardar resumen
        summary = f"""WCSR Evaluation - Independent Test Set
{'='*60}

Canciones evaluadas: {len(results['results'])} (NO usadas en entrenamiento)

System Configuration:
  - Beats per segment: 4
  - HMM: False
  
Results:
  - Global WCSR (weighted): {results['global_wcsr']:.4f} ({results['global_wcsr']*100:.2f}%)
  - Mean WCSR: {results['mean_wcsr']:.4f}
  - Median WCSR: {results['median_wcsr']:.4f}
  - Std Dev: {results['std_wcsr']:.4f}
  
Total Duration:
  - Total: {results['total_duration']:.1f}s ({results['total_duration']/60:.1f} min)
  - Correct: {results['total_correct']:.1f}s ({results['total_correct']/60:.1f} min)

Per-song results:
"""
        for r in sorted(results['results'], key=lambda x: x['wcsr'], reverse=True):
            summary += f"  {r['track_id']:30s}  WCSR={r['wcsr']:.3f}\n"
        
        (OUT / "wcsr_independent_summary.txt").write_text(summary)
        print(f"💾 Resumen guardado en: {OUT / 'wcsr_independent_summary.txt'}")
    
    print("\n✅ Evaluación completada!")

