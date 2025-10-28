# evaluate_wcsr_frames.py
"""
Script para evaluar WCSR comparando predicciones con ground truth.
"""

import numpy as np
import librosa
from pathlib import Path
import pandas as pd

# --- Config ---
SR = 11025
HOP = 512

# Mapeo de canciones en wcsr_test a archivos .lab
SONG_MAPPINGS = {
    "For you blue.mp3": "12_-_Let_It_Be/11_-_For_You_Blue.lab",
    "Love Me Do.mp3": "01_-_Please_Please_Me/08_-_Love_Me_Do.lab",
    "Misery.mp3": "01_-_Please_Please_Me/02_-_Misery.lab",
    "Please Please Me.mp3": "01_-_Please_Please_Me/07_-_Please_Please_Me.lab",
}

NOTES = ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"]
ENH = {"C#":"Db","D#":"Eb","F#":"Gb","G#":"Ab","A#":"Bb"}

def norm_label(lab: str) -> str:
    """Normaliza etiquetas de acordes."""
    if lab.upper() == "N": return "N"
    parts = lab.split(":")
    root = ENH.get(parts[0], parts[0]).title()
    qual = "min" if (len(parts)>1 and "min" in parts[1]) else "maj"
    return f"{root}:{qual}" if root in NOTES else "N"

def load_lab(path):
    """Carga archivo .lab con normalización."""
    rows = []
    with open(path, "r") as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 3:
                rows.append((float(p[0]), float(p[1]), norm_label(p[2])))
    rows.sort(key=lambda x: x[0])
    return rows  # [(t0,t1,label_norm),...]

def load_predicted_lab(path):
    """Carga archivo .lab con predicciones."""
    rows = []
    with open(path, "r") as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 3:
                rows.append((float(p[0]), float(p[1]), p[2]))
    rows.sort(key=lambda x: x[0])
    return rows

def label_at_time(t, intervals):
    """Devuelve el label del acorde que cubre el tiempo t."""
    for t0, t1, lab in intervals:
        if t >= t0 and t < t1:
            return lab
    return "N"

def compute_wcsr(pred_intervals, ref_intervals, exclude_N=True):
    """
    Calcula WCSR comparando intervalos predichos vs ground truth.
    
    Args:
        pred_intervals: [(t0, t1, label), ...] predicciones
        ref_intervals: [(t0, t1, label), ...] ground truth
        exclude_N: si True, excluye intervalos con label "N"
    
    Returns:
        wcsr: Weighted Chord Symbol Recall (0-1)
        correct_duration: duración correcta (segundos)
        total_duration: duración total evaluada (segundos)
    """
    # Flatten ambas secuencias a frames de 0.046s (HOP/SR)
    dt = HOP / SR  # ~0.0464 s
    total_duration = 0.0
    correct_duration = 0.0
    
    # Obtener duración máxima
    max_t = max([t1 for _, t1, _ in pred_intervals + ref_intervals])
    
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

def evaluate_all_songs(wcsr_test_dir="wcsr_test", outputs_dir="outputs"):
    """Evalúa todas las canciones y guarda métricas."""
    wcsr_path = Path(wcsr_test_dir)
    chordlab_path = Path("The Beatles Annotations/chordlab/The Beatles")
    outputs_path = Path(outputs_dir)
    
    results = []
    total_correct = 0.0
    total_duration = 0.0
    n_songs = 0
    
    # Procesar cada canción
    audio_files = sorted(wcsr_path.glob("*.mp3"))
    print(f"\n🎵 Evaluando {len(audio_files)} canciones...\n")
    
    for audio_file in audio_files:
        audio_name = audio_file.name
        print(f"▶ {audio_name}")
        
        if audio_name not in SONG_MAPPINGS:
            print(f"  ⚠️  Sin mapeo para: {audio_name}")
            continue
        
        lab_relative = SONG_MAPPINGS[audio_name]
        ref_lab_path = chordlab_path / lab_relative
        pred_lab_path = outputs_path / f"{audio_file.stem}_predicted.lab"
        
        if not ref_lab_path.exists():
            print(f"  ❌ Ground truth no existe: {ref_lab_path}")
            continue
        
        if not pred_lab_path.exists():
            print(f"  ❌ Predicciones no existen: {pred_lab_path}")
            continue
        
        try:
            # Cargar ground truth y predicciones
            ref_intervals = load_lab(ref_lab_path)
            pred_intervals = load_predicted_lab(pred_lab_path)
            
            # Calcular WCSR
            wcsr, correct, total = compute_wcsr(pred_intervals, ref_intervals, exclude_N=True)
            
            results.append({
                'song': audio_name,
                'wcsr': wcsr,
                'correct_duration': correct,
                'total_duration': total,
            })
            
            total_correct += correct
            total_duration += total
            n_songs += 1
            
            print(f"  WCSR: {wcsr:.4f} ({wcsr*100:.2f}%)")
            print(f"  Duración total: {total:.1f}s, correcta: {correct:.1f}s\n")
            
        except Exception as e:
            print(f"  ❌ Error: {e}\n")
            import traceback
            traceback.print_exc()
            continue
    
    # Métricas globales
    if n_songs > 0:
        global_wcsr = total_correct / total_duration if total_duration > 0 else 0.0
        mean_wcsr = np.mean([r['wcsr'] for r in results])
        median_wcsr = np.median([r['wcsr'] for r in results])
        std_wcsr = np.std([r['wcsr'] for r in results])
        
        print(f"\n{'='*60}")
        print(f"📊 RESULTADOS GLOBALES")
        print(f"{'='*60}")
        print(f"\nCanciones evaluadas: {n_songs}")
        print(f"WCSR Global (ponderado): {global_wcsr:.4f} ({global_wcsr*100:.2f}%)")
        print(f"WCSR Promedio:           {mean_wcsr:.4f} ({mean_wcsr*100:.2f}%)")
        print(f"WCSR Mediana:            {median_wcsr:.4f}")
        print(f"Desviación estándar:     {std_wcsr:.4f}")
        print(f"\nDuración total evaluada: {total_duration:.1f}s ({total_duration/60:.1f} min)")
        print(f"Duración correcta:      {total_correct:.1f}s ({total_correct/60:.1f} min)")
        print(f"\nResultados por canción:")
        for i, r in enumerate(sorted(results, key=lambda x: x['wcsr'], reverse=True), 1):
            print(f"  {i}. {r['song']:30s}  WCSR={r['wcsr']:.4f} ({r['wcsr']*100:.2f}%)")
        print(f"\n{'='*60}\n")
        
        # Guardar resultados
        results_df = pd.DataFrame(results)
        output_path = Path("analysis_out") / "wcsr_frames_results.csv"
        results_df.to_csv(output_path, index=False)
        print(f"💾 Resultados guardados en: {output_path}")
    
    return results

if __name__ == "__main__":
    evaluate_all_songs()

